// Raster coordinates refer to pixel centers (x+.5,y+.5); RLE is COCO
// column-major and starts with the length of the background run.
export const clone = (value) => structuredClone(value);
export const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
export function decodeRLE(rle) {
  const [h, w] = rle.size || [];
  if (
    !Number.isSafeInteger(h) ||
    !Number.isSafeInteger(w) ||
    h <= 0 ||
    w <= 0 ||
    h * w > 100_000_000 ||
    !Array.isArray(rle.counts)
  )
    throw Error("Unsupported or oversized mask.");
  const pixels = new Uint8Array(w * h);
  let offset = 0,
    fill = 0;
  for (const run of rle.counts) {
    if (!Number.isSafeInteger(run) || run < 0 || offset + run > pixels.length)
      throw Error("Invalid mask run length.");
    if (fill)
      for (let i = offset; i < offset + run; i++)
        pixels[(i % h) * w + Math.floor(i / h)] = 1;
    offset += run;
    fill = 1 - fill;
  }
  if (offset !== pixels.length)
    throw Error("Mask dimensions do not match its runs.");
  return { width: w, height: h, pixels };
}
export function encodeRLE({ width: w, height: h, pixels }) {
  const counts = [];
  let fill = 0,
    run = 0;
  for (let x = 0; x < w; x++)
    for (let y = 0; y < h; y++) {
      const v = pixels[y * w + x] ? 1 : 0;
      if (v === fill) run++;
      else {
        counts.push(run);
        run = 1;
        fill = v;
      }
    }
  counts.push(run);
  return { size: [h, w], counts };
}
export function pointInPolygon([x, y], points) {
  let inside = false;
  for (let i = 0, j = points.length - 1; i < points.length; j = i++) {
    const [a, b] = points[i],
      [c, d] = points[j];
    if (b > y !== d > y && x < ((c - a) * (y - b)) / (d - b) + a)
      inside = !inside;
  }
  return inside;
}
export function bbox(g) {
  if (g.type === "box") return g.xyxy.slice();
  if (g.type === "mask") {
    const [h, w] = g.rle.size;
    return [0, 0, w, h];
  }
  return [
    Math.min(...g.points.map((p) => p[0])),
    Math.min(...g.points.map((p) => p[1])),
    Math.max(...g.points.map((p) => p[0])),
    Math.max(...g.points.map((p) => p[1])),
  ];
}
export function rasterize(g, w, h) {
  if (g.type === "mask") return decodeRLE(g.rle);
  if (w * h > 100_000_000)
    throw Error("This image exceeds the interactive mask memory limit.");
  const m = { width: w, height: h, pixels: new Uint8Array(w * h) },
    b = bbox(g);
  for (
    let y = Math.max(0, Math.floor(b[1]));
    y < Math.min(h, Math.ceil(b[3]));
    y++
  )
    for (
      let x = Math.max(0, Math.floor(b[0]));
      x < Math.min(w, Math.ceil(b[2]));
      x++
    )
      if (g.type === "box" || pointInPolygon([x + 0.5, y + 0.5], g.points))
        m.pixels[y * w + x] = 1;
  return m;
}
export function brushLine(mask, a, b, radius, add = true) {
  const { width: w, height: h, pixels } = mask,
    dx = b[0] - a[0],
    dy = b[1] - a[1],
    length = dx * dx + dy * dy;
  const x0 = Math.max(0, Math.floor(Math.min(a[0], b[0]) - radius)),
    x1 = Math.min(w, Math.ceil(Math.max(a[0], b[0]) + radius)),
    y0 = Math.max(0, Math.floor(Math.min(a[1], b[1]) - radius)),
    y1 = Math.min(h, Math.ceil(Math.max(a[1], b[1]) + radius));
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++) {
      const t = length
        ? clamp(((x + 0.5 - a[0]) * dx + (y + 0.5 - a[1]) * dy) / length, 0, 1)
        : 0;
      const distance =
        (x + 0.5 - a[0] - t * dx) ** 2 + (y + 0.5 - a[1] - t * dy) ** 2;
      if (distance <= radius * radius) pixels[y * w + x] = add ? 1 : 0;
    }
  return [x0, y0, x1, y1];
}
export function updateMaskCanvas(canvas, mask, color, rect) {
  const [x0, y0, x1, y1] = rect,
    w = x1 - x0,
    h = y1 - y0;
  if (!w || !h) return;
  const ctx = canvas.getContext("2d"),
    data = ctx.createImageData(w, h),
    hex = /^#[0-9a-f]{6}$/i.test(color) ? color : "#80b76f",
    rgb = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  for (let y = y0; y < y1; y++)
    for (let x = x0; x < x1; x++)
      if (mask.pixels[y * mask.width + x])
        data.data.set([...rgb, 255], ((y - y0) * w + x - x0) * 4);
  ctx.putImageData(data, x0, y0);
}
export function nearestVertex(points, p, tolerance) {
  let found = -1,
    best = tolerance;
  points.forEach((q, i) => {
    const d = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (d < best) {
      best = d;
      found = i;
    }
  });
  return found;
}
export function nearestEdge(points, p, tolerance) {
  let best = null;
  points.forEach((a, i) => {
    const b = points[(i + 1) % points.length],
      dx = b[0] - a[0],
      dy = b[1] - a[1],
      den = dx * dx + dy * dy;
    if (!den) return;
    const t = clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / den, 0, 1),
      q = [a[0] + dx * t, a[1] + dy * t],
      distance = Math.hypot(q[0] - p[0], q[1] - p[1]);
    if (distance < tolerance && (!best || distance < best.distance))
      best = { index: i, point: q, distance };
  });
  return best;
}
export function polygonError(points) {
  if (points.length < 3) return "A polygon needs at least three vertices.";
  if (points.some((p) => !p.every(Number.isFinite)))
    return "Coordinates must be finite.";
  let area = 0;
  for (let i = 0; i < points.length; i++) {
    const a = points[i],
      b = points[(i + 1) % points.length];
    area += a[0] * b[1] - b[0] * a[1];
  }
  if (Math.abs(area) < 0.01) return "The polygon has no area.";
  const cross = (a, b, c) =>
    (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
  for (let i = 0; i < points.length; i++)
    for (let j = i + 2; j < points.length; j++) {
      if (i === 0 && j === points.length - 1) continue;
      const a = points[i],
        b = points[(i + 1) % points.length],
        c = points[j],
        d = points[(j + 1) % points.length];
      if (
        cross(a, b, c) * cross(a, b, d) < 0 &&
        cross(c, d, a) * cross(c, d, b) < 0
      )
        return "Polygon edges cross. Move or remove a vertex to repair the outline.";
    }
  return null;
}
export function maskCanvas(mask, color = "#80b76f") {
  const c = document.createElement("canvas");
  c.width = mask.width;
  c.height = mask.height;
  const ctx = c.getContext("2d"),
    img = ctx.createImageData(c.width, c.height);
  const hex = /^#[0-9a-f]{6}$/i.test(color) ? color : "#80b76f";
  const rgb = [1, 3, 5].map((i) => parseInt(hex.slice(i, i + 2), 16));
  for (let i = 0; i < mask.pixels.length; i++)
    if (mask.pixels[i]) {
      img.data.set([...rgb, 255], i * 4);
    }
  ctx.putImageData(img, 0, 0);
  return c;
}
