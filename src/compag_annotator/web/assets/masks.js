// Compressed canonical masks stay compressed. Only an actively brushed mask is
// rasterized by geometry.js. Display paths and exact hits share validated runs.
export class MaskStore {
  constructor(load, changed = () => {}) {
    this.load = load;
    this.changed = changed;
    this.entries = new Map();
    this.pending = new Map();
    this.queue = [];
    this.active = 0;
    this.epoch = 0;
  }
  clear() {
    this.epoch++;
    this.entries.clear();
    this.pending.clear();
  }
  async ensure(object, priority = false) {
    if (object.geometry?.type !== "mask" || object.geometry.rle?.counts)
      return object.geometry;
    const key = `${object.id}:${object.revision}:${object.geometry.ref || ""}`;
    const epoch = this.epoch;
    if (this.pending.has(key)) {
      if (priority) {
        const index = this.queue.findIndex(
          (task) => task.key === key && task.epoch === epoch,
        );
        if (index > 0) this.queue.unshift(...this.queue.splice(index, 1));
      }
      return this.pending.get(key);
    }
    const task = new Promise((resolve, reject) => {
      const run = async () => {
        try {
          if (epoch !== this.epoch) {
            resolve(null);
            return;
          }
          const result = await this.load(object);
          if (epoch !== this.epoch) {
            resolve(null);
            return;
          }
          object.geometry = result.geometry || result;
          resolve(object.geometry);
          this.changed();
        } catch (error) {
          reject(error);
        }
      };
      const queued = { key, epoch, run };
      priority ? this.queue.unshift(queued) : this.queue.push(queued);
      this.drain();
    });
    this.pending.set(key, task);
    return task;
  }
  drain() {
    while (this.active < 4 && this.queue.length) {
      this.active++;
      this.queue
        .shift().run()
        .finally(() => {
          this.active--;
          this.drain();
        });
    }
  }
  get(object) {
    const geometry = object.geometry;
    if (!geometry?.rle?.counts) return null;
    let entry = this.entries.get(geometry);
    if (entry) return entry;
    const [height, width] = geometry.rle.size;
    const ends = new Float64Array(geometry.rle.counts.length),
      path = new Path2D();
    let offset = 0,
      xmin = width,
      ymin = height,
      xmax = 0,
      ymax = 0;
    geometry.rle.counts.forEach((count, index) => {
      if (
        !Number.isSafeInteger(count) ||
        count < 0 ||
        offset + count > width * height
      )
        throw Error("Invalid canonical mask runs.");
      const end = offset + count;
      ends[index] = end;
      if (index % 2 && count) {
        let x = Math.floor(offset / height),
          y = offset % height;
        const stopX = Math.floor((end - 1) / height),
          stopY = ((end - 1) % height) + 1;
        xmin = Math.min(xmin, x);
        xmax = Math.max(xmax, stopX + 1);
        if (x === stopX) {
          path.rect(x, y, 1, stopY - y);
          ymin = Math.min(ymin, y);
          ymax = Math.max(ymax, stopY);
        } else {
          path.rect(x, y, 1, height - y);
          if (stopX > x + 1) path.rect(x + 1, 0, stopX - x - 1, height);
          path.rect(stopX, 0, 1, stopY);
          ymin = 0;
          ymax = height;
        }
      }
      offset = end;
    });
    if (offset !== width * height) throw Error("Canonical mask size mismatch.");
    entry = { ends, path, width, height, bbox: [xmin, ymin, xmax, ymax] };
    this.entries.set(geometry, entry);
    return entry;
  }
  hit(object, point) {
    const entry = this.get(object);
    if (!entry) return false;
    const x = Math.floor(point[0]),
      y = Math.floor(point[1]);
    if (x < 0 || y < 0 || x >= entry.width || y >= entry.height) return false;
    const pixel = x * entry.height + y;
    let lo = 0,
      hi = entry.ends.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (entry.ends[mid] <= pixel) lo = mid + 1;
      else hi = mid;
    }
    return lo % 2 === 1;
  }
}
