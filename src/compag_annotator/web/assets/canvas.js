import { MaskStore } from "./masks.js";
import {
  clone,
  clamp,
  encodeRLE,
  pointInPolygon,
  rasterize,
  brushLine,
  nearestVertex,
  nearestEdge,
  polygonError,
  maskCanvas,
  updateMaskCanvas,
  bbox,
} from "./geometry.js";
export class AnnotationCanvas {
  constructor(canvas, callbacks = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.callbacks = callbacks;
    this.objects = [];
    this.classes = [];
    this.selected = null;
    this.selection = new Set();
    this.tool = "select";
    this.opacity = 0.36;
    this.hiddenClasses = new Set();
    this.hiddenObjects = new Set();
    this.showProposals = true;
    this.showRejected = false;
    this.showSuperseded = false;
    this.showUnassigned = true;
    this.hideAssigned = false;
    this.hideGeneratedSAM = false;
    this.showMaskFill = true;
    this.reviewPendingOnly = false;
    this.previews = [];
    this.hiddenLayers = new Set();
    this.draft = [];
    this.maskCache = new MaskStore(
      (o) => this.callbacks.geometry?.(o),
      () => {
        if (this.alive) this.requestRender();
      },
    );
    this.transform = { x: 0, y: 0, scale: 1 };
    this.points = [];
    this.labels = [];
    this.brushRadius = 15;
    this.space = false;
    this.locked = false;
    this.alive = true;
    this.events = new AbortController();
    const opts = { signal: this.events.signal };
    canvas.addEventListener(
      "pointerdown",
      (e) =>
        this.down(e).catch((error) => this.callbacks.error?.(error.message)),
      opts,
    );
    canvas.addEventListener(
      "pointerleave",
      () => {
        this.hover = null;
        this.render();
      },
      opts,
    );
    canvas.addEventListener("pointermove", (e) => this.move(e), opts);
    canvas.addEventListener("pointerup", (e) => this.up(e), opts);
    canvas.addEventListener("pointercancel", () => this.cancelGesture(), opts);
    canvas.addEventListener("dblclick", (e) => this.doubleClick(e), opts);
    canvas.addEventListener("contextmenu", (e) => e.preventDefault(), opts);
    canvas.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        this.zoom(Math.exp(-e.deltaY * 0.0015), this.screen(e));
      },
      { ...opts, passive: false },
    );
    window.addEventListener(
      "keydown",
      (e) => {
        if (e.target.closest("input,textarea,select,dialog")) return;
        if (e.code === "Space") {
          e.preventDefault();
          this.space = true;
        }
        if (e.key === "Escape") {
          this.cancelGesture();
          this.draft = [];
          this.render();
        }
        if (e.key === "Enter" && this.draft.length) {
          e.preventDefault();
          this.finishPolygon();
        }
      },
      opts,
    );
    window.addEventListener(
      "keyup",
      (e) => {
        if (e.code === "Space") this.space = false;
      },
      opts,
    );
    window.addEventListener(
      "blur",
      () => {
        this.space = false;
        this.cancelGesture();
      },
      opts,
    );
    this.observer = new ResizeObserver(() => this.resize());
    this.observer.observe(canvas);
    this.resize();
  }
  destroy() {
    this.alive = false;
    if (this.renderFrame != null) cancelAnimationFrame(this.renderFrame);
    this.renderFrame = null;
    this.events.abort();
    this.observer.disconnect();
    this.maskCache.clear();
  }
  resize() {
    const r = this.canvas.getBoundingClientRect(),
      d = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.round(r.width * d));
    this.canvas.height = Math.max(1, Math.round(r.height * d));
    this.dpr = d;
    this.render();
  }
  async setImage(record, url) {
    const seq = (this.loadSeq = (this.loadSeq || 0) + 1);
    this.image = null;
    this.record = record;
    this.objects = [];
    this.selected = null;
    this.selection = new Set();
    this.draft = [];
    this.maskCache.clear();
    this.render();
    const image = new Image();
    image.src = url;
    await image.decode();
    if (!this.alive || seq !== this.loadSeq) return;
    this.image = image;
    this.fit();
  }
  fit() {
    if (!this.record) return;
    const r = this.canvas.getBoundingClientRect();
    const scale = Math.min(
      (r.width - 50) / this.record.width,
      (r.height - 50) / this.record.height,
    );
    this.transform = {
      scale: Math.max(0.01, scale),
      x: (r.width - this.record.width * scale) / 2,
      y: (r.height - this.record.height * scale) / 2,
    };
    this.render();
  }
  screen(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX - r.left, e.clientY - r.top];
  }
  original(e) {
    const [x, y] = this.screen(e),
      t = this.transform;
    return [(x - t.x) / t.scale, (y - t.y) / t.scale];
  }
  bounded(p) {
    return [
      clamp(p[0], 0, this.record.width),
      clamp(p[1], 0, this.record.height),
    ];
  }
  inside(p) {
    return (
      this.record &&
      p[0] >= 0 &&
      p[1] >= 0 &&
      p[0] < this.record.width &&
      p[1] < this.record.height
    );
  }
  zoom(factor, point) {
    const t = this.transform,
      p = point || [this.canvas.clientWidth / 2, this.canvas.clientHeight / 2],
      next = clamp(t.scale * factor, 0.01, 40);
    t.x = p[0] - ((p[0] - t.x) * next) / t.scale;
    t.y = p[1] - ((p[1] - t.y) * next) / t.scale;
    t.scale = next;
    this.render();
  }
  setTool(tool) {
    this.tool = tool;
    this.draft = [];
    this.cancelGesture();
    this.render();
  }
  visible(o) {
    return (
      !this.hiddenObjects.has(o.id) &&
      !this.hiddenClasses.has(o.class_id) &&
      (this.showProposals || o.status !== "proposal") &&
      (this.showRejected || o.status !== "rejected") &&
      (this.showSuperseded || o.status !== "superseded") &&
      (this.showUnassigned || !!o.class_id) &&
      (!this.hideAssigned || !o.class_id) &&
      (!this.hideGeneratedSAM || !(o.layer_id || o.source?.operation === "automatic_mask_generation")) &&
      (!this.reviewPendingOnly || ["draft", "proposal"].includes(o.status)) &&
      !this.hiddenLayers.has(o.layer_id)
    );
  }
  pruneHiddenSelection() {
    const visibleIds = new Set(this.objects.filter((o) => this.visible(o)).map((o) => o.id));
    this.selection = new Set([...this.selection].filter((id) => visibleIds.has(id)));
    if (!visibleIds.has(this.selected))
      this.selected = [...this.selection].at(-1) || null;
    if (!visibleIds.has(this.hover)) this.hover = null;
    this.overlaps = (this.overlaps || []).filter((id) => visibleIds.has(id));
    this.hoverSeq = (this.hoverSeq || 0) + 1;
  }
  color(o) {
    return this.maskColors?.get(o.id) || this.defaultMaskColor || this.classes.find((c) => c.id === o.class_id)?.color || "#aebcc4";
  }
  bounds(o) {
    if (o.geometry?.type === "mask")
      return (
        this.maskCache.get(o)?.bbox ||
        o.bbox ||
        o.geometry.bbox || [0, 0, this.record.width, this.record.height]
      );
    return bbox(o.geometry);
  }
  near(o, p) {
    const b = this.bounds(o);
    return p[0] >= b[0] && p[0] < b[2] && p[1] >= b[1] && p[1] < b[3];
  }
  async hits(p) {
    const candidates = this.objects.filter(
      (o) =>
        this.visible(o) &&
        !["rejected", "superseded"].includes(o.status) &&
        this.near(o, p),
    );
    await Promise.all(candidates.map((o) => this.maskCache.ensure(o, true)));
    return candidates.filter((o) => this.visible(o) && this.hit(o, p)).reverse();
  }
  async hoverAt(p) {
    const seq = (this.hoverSeq = (this.hoverSeq || 0) + 1);
    try {
      const hits = await this.hits(p);
      if (!this.alive || seq !== this.hoverSeq || this.gesture) return;
      this.overlaps = hits.map((o) => o.id);
      const id = hits[0]?.id;
      if (id !== this.hover) {
        this.hover = id;
        this.render();
      }
    } catch (error) {
      this.callbacks.error?.(error.message);
    }
  }
  hit(o, p) {
    const g = o.geometry;
    if (!g) return false;
    if (g.type === "box") {
      const b = g.xyxy;
      return p[0] >= b[0] && p[0] <= b[2] && p[1] >= b[1] && p[1] <= b[3];
    }
    if (g.type === "polygon") return pointInPolygon(p, g.points);
    if (g.type === "mask") return this.maskCache.hit(o, p);
    return false;
  }
  select(id, additive = false) {
    if (id && !this.objects.some((o) => o.id === id && this.visible(o))) return;
    if (additive && id) {
      this.selection.has(id)
        ? this.selection.delete(id)
        : this.selection.add(id);
      this.selected = this.selection.has(id)
        ? id
        : [...this.selection].at(-1) || null;
    } else {
      this.selected = id;
      this.selection = new Set(id ? [id] : []);
    }
    this.callbacks.select?.(this.selected);
    this.render();
  }
  async down(e) {
    if (!this.image) return;
    this.canvas.focus();
    const p = this.original(e),
      s = this.screen(e);
    this.canvas.setPointerCapture(e.pointerId);
    if (this.space || this.tool === "pan" || e.button === 1 || e.button === 2) {
      this.gesture = {
        kind: "pan",
        start: s,
        transform: { ...this.transform },
      };
      return;
    }
    if (this.locked || !this.inside(p)) return;
    if (["positive", "negative"].includes(this.tool)) {
      this.callbacks.prompt?.(p, this.tool === "positive" ? 1 : 0);
      return;
    }
    if (this.tool === "polygon") {
      if (
        this.draft.length >= 3 &&
        Math.hypot(p[0] - this.draft[0][0], p[1] - this.draft[0][1]) *
          this.transform.scale <
          9
      ) {
        this.finishPolygon();
        return;
      }
      if (
        !this.draft.length ||
        Math.hypot(p[0] - this.draft.at(-1)[0], p[1] - this.draft.at(-1)[1]) >
          0.1
      )
        this.draft.push(p);
      this.render();
      return;
    }
    if (["box", "prompt-box"].includes(this.tool)) {
      this.gesture = { kind: this.tool, start: p, end: p };
      this.render();
      return;
    }
    const selected = this.objects.find((o) => o.id === this.selected);
    if (this.tool === "brush-add" || this.tool === "brush-subtract") {
      if (selected && selected.geometry.type !== "mask") {
        this.callbacks.error?.(
          "Use “Convert to mask” before brushing a polygon or box.",
        );
        return;
      }
      if (!selected && this.tool === "brush-subtract") {
        this.callbacks.error?.("Select a mask before subtracting pixels.");
        return;
      }
      if (selected) await this.maskCache.ensure(selected, true);
      const mask = selected
        ? rasterize(selected.geometry, this.record.width, this.record.height)
        : rasterize(
            { type: "polygon", points: [] },
            this.record.width,
            this.record.height,
          );
      this.gesture = {
        kind: "brush",
        object: selected,
        mask,
        last: p,
        add: this.tool === "brush-add",
      };
      brushLine(mask, p, p, this.brushRadius, this.gesture.add);
      this.gesture.canvas = maskCanvas(mask, this.color(selected || {}));
      this.render();
      return;
    }
    if (this.tool === "select" || this.tool === "paint") {
      const hits = await this.hits(p);
      if (this.locked || !this.alive) return;
      let hit = hits[0];
      if (e.altKey && hits.length > 1)
        hit =
          hits[
            (hits.findIndex((o) => o.id === this.selected) + 1) % hits.length
          ];
      this.overlaps = hits.map((o) => o.id);
      this.select(
        hit?.id || null,
        this.tool === "select" && (e.shiftKey || e.ctrlKey || e.metaKey),
      );
      if (hit && this.tool === "paint") await this.callbacks.paint?.(hit);
      return;
    }
    if (this.tool !== "edit") return;
    if (selected?.geometry.type === "polygon") {
      const idx = nearestVertex(
        selected.geometry.points,
        p,
        9 / this.transform.scale,
      );
      if (idx >= 0) {
        if (e.shiftKey) {
          if (selected.geometry.points.length <= 3) {
            this.callbacks.error?.("Keep at least three vertices.");
            return;
          }
          const g = clone(selected.geometry);
          g.points.splice(idx, 1);
          this.commit(selected, g);
          return;
        }
        this.gesture = {
          kind: "vertex",
          object: selected,
          before: clone(selected.geometry),
          index: idx,
          start: p,
        };
        return;
      }
    }
    if (
      selected?.geometry.type === "polygon" &&
      nearestEdge(selected.geometry.points, p, 10 / this.transform.scale)
    ) {
      this.gesture = {
        kind: "move",
        object: selected,
        before: clone(selected.geometry),
        start: p,
      };
      return;
    }
    if (selected?.geometry.type === "box") {
      const b = selected.geometry.xyxy,
        pts = [
          [b[0], b[1]],
          [b[2], b[1]],
          [b[2], b[3]],
          [b[0], b[3]],
        ],
        idx = nearestVertex(pts, p, 10 / this.transform.scale);
      if (idx >= 0) {
        this.gesture = {
          kind: "resize",
          object: selected,
          before: clone(selected.geometry),
          index: idx,
          start: p,
        };
        return;
      }
    }
    const hits = this.objects
      .filter((o) => this.visible(o) && this.hit(o, p))
      .reverse();
    let hit = hits[0];
    if (e.altKey && hits.length > 1)
      hit =
        hits[(hits.findIndex((o) => o.id === this.selected) + 1) % hits.length];
    this.select(hit?.id || null);
    if (hit && hit.geometry.type !== "mask")
      this.gesture = {
        kind: "move",
        object: hit,
        before: clone(hit.geometry),
        start: p,
      };
  }
  move(e) {
    this.pointer = this.original(e);
    const g = this.gesture;
    if (!g) {
      this.callbacks.view?.(this.transform.scale, this.pointer);
      if (["select", "paint", "edit"].includes(this.tool) && !this.locked)
        this.hoverAt(this.pointer);
      else if (this.draft.length || this.tool.startsWith("brush"))
        this.requestRender();
      return;
    }
    if (g.kind === "pan") {
      const s = this.screen(e);
      this.transform.x = g.transform.x + s[0] - g.start[0];
      this.transform.y = g.transform.y + s[1] - g.start[1];
    } else {
      const p = this.bounded(this.pointer);
      if (g.kind === "box" || g.kind === "prompt-box") g.end = p;
      else if (g.kind === "brush") {
        const dirty = brushLine(g.mask, g.last, p, this.brushRadius, g.add);
        updateMaskCanvas(g.canvas, g.mask, this.color(g.object || {}), dirty);
        g.last = p;
      } else if (g.kind === "vertex") {
        const geometry = clone(g.before);
        geometry.points[g.index] = p;
        g.preview = geometry;
      } else if (g.kind === "resize") {
        const b = g.before.xyxy.slice(),
          opposite = [
            [b[2], b[3]],
            [b[0], b[3]],
            [b[0], b[1]],
            [b[2], b[1]],
          ][g.index];
        g.preview = {
          type: "box",
          xyxy: [
            Math.min(p[0], opposite[0]),
            Math.min(p[1], opposite[1]),
            Math.max(p[0], opposite[0]),
            Math.max(p[1], opposite[1]),
          ],
        };
      } else if (g.kind === "move") {
        const b = bbox(g.before),
          dx = clamp(p[0] - g.start[0], -b[0], this.record.width - b[2]),
          dy = clamp(p[1] - g.start[1], -b[1], this.record.height - b[3]);
        g.preview =
          g.before.type === "box"
            ? {
                type: "box",
                xyxy: g.before.xyxy.map((v, i) => v + (i % 2 ? dy : dx)),
              }
            : {
                type: "polygon",
                points: g.before.points.map((q) => [q[0] + dx, q[1] + dy]),
              };
      }
    }
    this.requestRender();
  }
  up(e) {
    if (this.canvas.hasPointerCapture(e.pointerId))
      this.canvas.releasePointerCapture(e.pointerId);
    const g = this.gesture;
    this.gesture = null;
    if (!g) return;
    if (g.kind === "box" || g.kind === "prompt-box") {
      const b = [
        Math.min(g.start[0], g.end[0]),
        Math.min(g.start[1], g.end[1]),
        Math.max(g.start[0], g.end[0]),
        Math.max(g.start[1], g.end[1]),
      ];
      if (b[2] - b[0] >= 1 && b[3] - b[1] >= 1) {
        if (g.kind === "prompt-box") this.callbacks.boxPrompt?.(b);
        else this.commit(null, { type: "box", xyxy: b });
      }
    } else if (g.kind === "brush") {
      if (g.mask.pixels.some(Boolean))
        this.commit(g.object, { type: "mask", rle: encodeRLE(g.mask) });
      else
        this.callbacks.error?.(
          "The brush would leave an empty mask. Delete the object explicitly instead.",
        );
    } else if (
      g.preview &&
      JSON.stringify(g.preview) !== JSON.stringify(g.before)
    )
      this.commit(g.object, g.preview);
    this.render();
  }
  cancelGesture() {
    this.gesture = null;
    this.render();
  }
  doubleClick(e) {
    if (this.locked) return;
    if (this.tool === "polygon") {
      this.finishPolygon();
      return;
    }
    const o = this.objects.find((o) => o.id === this.selected);
    if (this.tool === "edit" && o?.geometry.type === "polygon") {
      const edge = nearestEdge(
        o.geometry.points,
        this.original(e),
        10 / this.transform.scale,
      );
      if (edge) {
        const g = clone(o.geometry);
        g.points.splice(edge.index + 1, 0, edge.point);
        this.commit(o, g);
      }
    }
  }
  finishPolygon() {
    if (this.locked) return;
    const error = polygonError(this.draft);
    if (error) {
      this.callbacks.error?.(error);
      return;
    }
    const points = this.draft.map((p) => p.slice());
    this.draft = [];
    this.commit(null, { type: "polygon", points });
    this.render();
  }
  commit(object, geometry) {
    if (geometry.type === "polygon") {
      const error = polygonError(geometry.points);
      if (error) {
        this.callbacks.error?.(error);
        return;
      }
    }
    this.callbacks.change?.(object, geometry);
  }
  drawGeometry(o, preview = false) {
    const c = this.ctx,
      g = o.geometry;
    if (!g) return;
    if (g.type === "mask" && !g.rle?.counts) {
      if (this.callbacks.geometry)
        this.maskCache
          .ensure(o, this.selection.has(o.id) || this.selected === o.id)
          .catch((error) => this.callbacks.error?.(error.message));
      return;
    }
    const color = preview ? "#f9d779" : this.color(o),
      fillEnabled = preview || this.showMaskFill !== false,
      selected = this.selection.has(o.id) || o.id === this.selected,
      hovered = o.id === this.hover;
    c.save();
    c.strokeStyle = selected ? "#fff" : color;
    c.fillStyle = color;
    c.lineWidth = (selected ? 2.5 : 1.4) / this.transform.scale;
    c.setLineDash(
      o.status === "proposal"
        ? [5 / this.transform.scale, 4 / this.transform.scale]
        : [],
    );
    if (g.type === "mask" && g.rle?.counts) {
      const entry = this.maskCache.get(o);
      c.globalAlpha =
        selected || hovered
          ? Math.min(1, this.opacity + (selected ? 0.3 : 0.18))
          : this.opacity;
      if (fillEnabled) c.fill(entry.path);
      if (selected || hovered) {
        c.globalAlpha = 1;
        c.strokeStyle = selected ? "#fff" : "#f9d779";
        const b = entry.bbox;
        c.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      }
    } else {
      c.beginPath();
      if (g.type === "box")
        c.rect(
          g.xyxy[0],
          g.xyxy[1],
          g.xyxy[2] - g.xyxy[0],
          g.xyxy[3] - g.xyxy[1],
        );
      else if (g.type === "polygon" && g.points.length) {
        c.moveTo(...g.points[0]);
        g.points.slice(1).forEach((p) => c.lineTo(...p));
        c.closePath();
      }
      c.globalAlpha = this.opacity;
      if (fillEnabled) c.fill();
      c.globalAlpha = 1;
      c.stroke();
      if (selected && this.tool === "edit") {
        const b = g.xyxy,
          points =
            g.type === "polygon"
              ? g.points
              : [
                  [b[0], b[1]],
                  [b[2], b[1]],
                  [b[2], b[3]],
                  [b[0], b[3]],
                ];
        for (const p of points) {
          c.beginPath();
          c.arc(...p, 4 / this.transform.scale, 0, Math.PI * 2);
          c.fillStyle = "#23392d";
          c.fill();
          c.strokeStyle = "#fff";
          c.stroke();
        }
      }
    }
    c.restore();
  }
  inView(o) {
    const b = this.bounds(o),
      t = this.transform;
    return (
      b[2] * t.scale + t.x >= 0 &&
      b[3] * t.scale + t.y >= 0 &&
      b[0] * t.scale + t.x <= this.canvas.clientWidth &&
      b[1] * t.scale + t.y <= this.canvas.clientHeight
    );
  }
  requestRender() {
    if (!this.alive || this.renderFrame != null) return;
    this.renderFrame = requestAnimationFrame(() => {
      this.renderFrame = null;
      if (this.alive) this.render();
    });
  }
  render() {
    if (this.renderFrame != null) {
      cancelAnimationFrame(this.renderFrame);
      this.renderFrame = null;
    }
    if (!this.ctx) return;
    const c = this.ctx,
      d = this.dpr || 1,
      t = this.transform;
    c.setTransform(d, 0, 0, d, 0, 0);
    c.clearRect(0, 0, this.canvas.width / d, this.canvas.height / d);
    if (!this.image) return;
    c.translate(t.x, t.y);
    c.scale(t.scale, t.scale);
    c.imageSmoothingEnabled = t.scale < 2;
    c.drawImage(this.image, 0, 0, this.record.width, this.record.height);
    c.save();
    c.beginPath();
    c.rect(0, 0, this.record.width, this.record.height);
    c.clip();
    for (const o of this.objects)
      if (
        this.visible(o) &&
        this.inView(o) &&
        !(this.gesture?.kind === "brush" && this.gesture.object?.id === o.id)
      )
        this.drawGeometry(
          this.gesture?.object === o && this.gesture.preview
            ? { ...o, geometry: this.gesture.preview }
            : o,
        );
    if (this.preview) this.drawGeometry(this.preview, true);
    for (const preview of this.previews) this.drawGeometry(preview, true);
    const g = this.gesture;
    if (g?.kind === "brush") {
      c.save();
      c.globalAlpha = this.opacity;
      c.drawImage(g.canvas, 0, 0);
      c.restore();
    }
    if (g?.kind === "box" || g?.kind === "prompt-box") {
      c.strokeStyle = "#e4ffd7";
      c.lineWidth = 2 / t.scale;
      c.strokeRect(
        g.start[0],
        g.start[1],
        g.end[0] - g.start[0],
        g.end[1] - g.start[1],
      );
    }
    if (this.promptBox) {
      const b = this.promptBox;
      c.strokeStyle = "#f5d677";
      c.lineWidth = 2 / t.scale;
      c.setLineDash([6 / t.scale, 4 / t.scale]);
      c.strokeRect(b[0], b[1], b[2] - b[0], b[3] - b[1]);
      c.setLineDash([]);
    }
    if (this.draft.length) {
      c.strokeStyle = "#e4ffd7";
      c.lineWidth = 2 / t.scale;
      c.beginPath();
      c.moveTo(...this.draft[0]);
      this.draft.slice(1).forEach((p) => c.lineTo(...p));
      if (this.pointer) c.lineTo(...this.pointer);
      c.stroke();
      for (const p of this.draft) {
        c.beginPath();
        c.arc(...p, 3 / t.scale, 0, Math.PI * 2);
        c.fillStyle = "#d0edbc";
        c.fill();
      }
    }
    this.points.forEach((p, i) => {
      c.beginPath();
      c.arc(...p, 6 / t.scale, 0, Math.PI * 2);
      c.fillStyle = this.labels[i] ? "#a9ed95" : "#ff8e85";
      c.fill();
      c.fillStyle = "#192a20";
      c.font = `${11 / t.scale}px sans-serif`;
      c.textAlign = "center";
      c.textBaseline = "middle";
      c.fillText(this.numberedPoints ? String(i + 1) : this.labels[i] ? "+" : "−", ...p);
    });
    if (this.pointer && this.tool.startsWith("brush")) {
      c.beginPath();
      c.arc(...this.pointer, this.brushRadius, 0, Math.PI * 2);
      c.strokeStyle = "#fff";
      c.lineWidth = 1 / t.scale;
      c.stroke();
    }
    c.restore();
    this.callbacks.view?.(t.scale, this.pointer);
  }
}
