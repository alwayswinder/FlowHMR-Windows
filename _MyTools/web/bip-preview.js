import { BufferAttribute, BufferGeometry, DynamicDrawUsage, Group, LineBasicMaterial, LineSegments, Points, PointsMaterial } from "three";

export function createBipPreview(data, frames, fps) {
  const { version, parents, names, samples_per_frame: rate } = data;
  if (version !== 1 || data.frames !== frames || data.fps !== fps || rate !== 4 ||
      !Array.isArray(parents) || parents.length < 45 || parents.length > 128 ||
      names?.length !== parents.length || parents[0] !== -1 ||
      !parents.slice(1).every((parent, i) => Number.isInteger(parent) && parent >= 0 && parent <= i)) {
    throw new Error("Invalid Biped preview timeline or hierarchy");
  }
  const bytes = Uint8Array.from(atob(data.samples), ch => ch.charCodeAt(0));
  const stride = parents.length * 6;
  const count = (frames - 1) * rate + 1;
  if (bytes.byteLength !== count * stride * 4) throw new Error("Incomplete Biped preview");
  const samples = new Float32Array(bytes.buffer);
  if (!samples.every(Number.isFinite)) throw new Error("Invalid Biped preview coordinates");
  const axisNodes = names.flatMap((name, i) => / (Thigh|Calf|Foot|UpperArm|Forearm|Hand)$/.test(name) ? [i] : []);
  const attribute = (length) => new BufferAttribute(new Float32Array(length), 3).setUsage(DynamicDrawUsage);
  const dots = attribute(parents.length * 3), links = attribute((parents.length - 1) * 6), axes = attribute(axisNodes.length * 6);
  const geometry = (pos) => new BufferGeometry().setAttribute("position", pos);
  const bones = new LineSegments(geometry(links), new LineBasicMaterial({ color: 0xffb860 }));
  const joints = new Points(geometry(dots), new PointsMaterial({ color: 0xffe8a9, size: 5, sizeAttenuation: false }));
  const twists = new LineSegments(geometry(axes), new LineBasicMaterial({ color: 0x62cbd2 }));
  const group = new Group();
  group.name = "biped-preview";
  group.add(bones, joints, twists);
  for (const item of group.children) item.frustumCulled = false;
  return {
    group,
    draw(frame) {
      const time = Math.max(0, Math.min(count - 1, frame * rate));
      const lo = Math.floor(time), hi = Math.min(lo + 1, count - 1), weight = time - lo;
      const get = (i) => samples[lo * stride + i] * (1 - weight) + samples[hi * stride + i] * weight;
      for (let i = 0; i < parents.length; i++) {
        for (let c = 0; c < 3; c++) dots.array[i * 3 + c] = get(i * 6 + c);
        if (i === 0) continue;
        const start = (i - 1) * 6;
        for (let c = 0; c < 3; c++) {
          links.array[start + c] = get(parents[i] * 6 + c);
          links.array[start + 3 + c] = dots.array[i * 3 + c];
        }
      }
      for (const [i, node] of axisNodes.entries()) for (let c = 0; c < 6; c++) axes.array[i * 6 + c] = get(node * 6 + c);
      dots.needsUpdate = links.needsUpdate = axes.needsUpdate = true;
    },
    dispose() {
      for (const object of group.children) {
        object.geometry.dispose();
        object.material.dispose();
      }
    },
  };
}
