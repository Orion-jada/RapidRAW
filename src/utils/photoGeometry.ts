// Reads the buffer from `generate_photo_geometry` and turns it into what the depth, normals and 3D
// view draw. The backend's points and normals are in camera space: x right, y down, z forward.

export interface PhotoGeometry {
  width: number;
  height: number;
  points: Float32Array;
  normals: Float32Array;
  mask: Uint8Array;
  texture: { width: number; height: number; data: Uint8Array };
}

// How the editor shows the photo: quarter turns clockwise, then flips.
export interface MapOrientation {
  steps: number;
  flipHorizontal: boolean;
  flipVertical: boolean;
}

export interface PhotoMesh {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  // Distance from the camera to the middle of the scene, along its view.
  focusDistance: number;
  farDistance: number;
  // Focal length in pixels of the geometry grid.
  focalLength: number;
}

// Neighbouring pixels whose depths differ by more than this fraction aren't joined, so objects
// don't get stretched sheets between them and what's behind.
const DEPTH_EDGE_TOLERANCE = 0.04;
// Cosine of the steepest angle between a triangle and the camera's line of sight that's kept.
const MIN_FACING = 0.1;

export function parsePhotoGeometry(buffer: ArrayBuffer): PhotoGeometry {
  const header = new DataView(buffer, 0, 16);
  const width = header.getUint32(0, true);
  const height = header.getUint32(4, true);
  const textureWidth = header.getUint32(8, true);
  const textureHeight = header.getUint32(12, true);
  const pixels = width * height;

  let offset = 16;
  const points = new Float32Array(buffer, offset, pixels * 3);
  offset += pixels * 12;
  const normals = new Float32Array(buffer, offset, pixels * 3);
  offset += pixels * 12;
  const mask = new Uint8Array(buffer, offset, pixels);
  offset += pixels;
  const data = new Uint8Array(buffer, offset, textureWidth * textureHeight * 4);

  return { width, height, points, normals, mask, texture: { width: textureWidth, height: textureHeight, data } };
}

export function orientationOf(adjustments: {
  orientationSteps?: number;
  flipHorizontal?: boolean;
  flipVertical?: boolean;
}): MapOrientation {
  return {
    steps: (((adjustments.orientationSteps ?? 0) % 4) + 4) % 4,
    flipHorizontal: !!adjustments.flipHorizontal,
    flipVertical: !!adjustments.flipVertical,
  };
}

// The x and y of a normal-map vector (x right, y up) once the photo is turned and flipped.
function orientVector(x: number, y: number, orientation: MapOrientation): [number, number] {
  for (let i = 0; i < orientation.steps; i++) {
    [x, y] = [y, -x];
  }
  return [orientation.flipHorizontal ? -x : x, orientation.flipVertical ? -y : y];
}

// The normal map in the usual colors: red right, green up, blue toward the camera. Pixels stay
// where they are in the photo; only the directions are turned, so draw it with `drawOriented`.
export function normalMapImageData(geometry: PhotoGeometry, orientation: MapOrientation): ImageData {
  const { width, height, normals } = geometry;
  const image = new ImageData(width, height);
  for (let i = 0; i < width * height; i++) {
    let x = normals[i * 3];
    let y = -normals[i * 3 + 1];
    let z = -normals[i * 3 + 2];
    const length = Math.hypot(x, y, z);
    if (Number.isFinite(length) && length > 1e-6) {
      [x, y] = orientVector(x / length, y / length, orientation);
      z /= length;
    } else {
      [x, y, z] = [0, 0, 1];
    }
    image.data[i * 4] = Math.round((x * 0.5 + 0.5) * 255);
    image.data[i * 4 + 1] = Math.round((y * 0.5 + 0.5) * 255);
    image.data[i * 4 + 2] = Math.round((z * 0.5 + 0.5) * 255);
    image.data[i * 4 + 3] = 255;
  }
  return image;
}

// Draws `source` onto `canvas` turned and flipped like the photo in the editor, resizing the
// canvas to fit.
export function drawOriented(
  canvas: HTMLCanvasElement,
  source: CanvasImageSource,
  sourceWidth: number,
  sourceHeight: number,
  orientation: MapOrientation,
) {
  const sideways = orientation.steps % 2 === 1;
  canvas.width = sideways ? sourceHeight : sourceWidth;
  canvas.height = sideways ? sourceWidth : sourceHeight;
  const context = canvas.getContext('2d');
  if (!context) return;
  context.translate(canvas.width / 2, canvas.height / 2);
  context.scale(orientation.flipHorizontal ? -1 : 1, orientation.flipVertical ? -1 : 1);
  context.rotate((orientation.steps * Math.PI) / 2);
  context.drawImage(source, -sourceWidth / 2, -sourceHeight / 2);
}

// Google's polynomial fit of the Turbo color map (Apache-2.0), lowest power first.
const TURBO = [
  [0.13572138, 4.6153926, -42.66032258, 132.13108234, -152.94239396, 59.28637943],
  [0.09140261, 2.19418839, 4.84296658, -14.18503333, 4.27729857, 2.82956604],
  [0.1066733, 12.64194608, -60.58204836, 110.36276771, -89.90310912, 27.34824973],
];

// Colors a grayscale depth map with Turbo, from blue (far) to red (near).
export function colorizeDepth(image: ImageData): ImageData {
  const lookup = TURBO.map((coefficients) =>
    Array.from({ length: 256 }, (_, level) => {
      const t = level / 255;
      const value = coefficients.reduceRight((sum, c) => sum * t + c, 0);
      return Math.round(Math.min(Math.max(value, 0), 1) * 255);
    }),
  );
  const out = new ImageData(image.width, image.height);
  for (let i = 0; i < image.data.length; i += 4) {
    const level = image.data[i];
    out.data[i] = lookup[0][level];
    out.data[i + 1] = lookup[1][level];
    out.data[i + 2] = lookup[2][level];
    out.data[i + 3] = 255;
  }
  return out;
}

// A mesh with a vertex per pixel, in three.js camera space (x right, y up, looking down -z), with
// the photo's camera at the origin. Pixels without a surface and depth edges are left open.
export function buildPhotoMesh(geometry: PhotoGeometry): PhotoMesh {
  const { width, height, points, normals, mask } = geometry;
  const pixels = width * height;
  const positions = new Float32Array(pixels * 3);
  const meshNormals = new Float32Array(pixels * 3);
  const uvs = new Float32Array(pixels * 2);
  const depth = new Float32Array(pixels);
  const valid = new Uint8Array(pixels);

  let focalNumerator = 0;
  let focalDenominator = 0;
  const depths: number[] = [];

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      const px = points[i * 3];
      const py = points[i * 3 + 1];
      const pz = points[i * 3 + 2];
      uvs[i * 2] = (x + 0.5) / width;
      uvs[i * 2 + 1] = (y + 0.5) / height;

      const nx = normals[i * 3];
      const ny = normals[i * 3 + 1];
      const nz = normals[i * 3 + 2];
      const length = Math.hypot(nx, ny, nz);
      if (Number.isFinite(length) && length > 1e-6) {
        meshNormals[i * 3] = nx / length;
        meshNormals[i * 3 + 1] = -ny / length;
        meshNormals[i * 3 + 2] = -nz / length;
      } else {
        meshNormals[i * 3 + 2] = 1;
      }

      if (mask[i] < 128 || !(pz > 0) || !Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) {
        continue;
      }
      valid[i] = 1;
      depth[i] = pz;
      positions[i * 3] = px;
      positions[i * 3 + 1] = -py;
      positions[i * 3 + 2] = -pz;

      if (x % 4 === 0 && y % 4 === 0) {
        // The camera's focal length, fitted to pixel offset = f * point / depth.
        const u = x + 0.5 - width / 2;
        const v = y + 0.5 - height / 2;
        const a = px / pz;
        const b = py / pz;
        focalNumerator += u * a + v * b;
        focalDenominator += a * a + b * b;
        depths.push(pz);
      }
    }
  }

  // MoGe blends depth across an object's outline over a few pixels, so outlines are found from
  // each pixel's 3x3 neighbourhood rather than from neighbours alone.
  const edge = new Uint8Array(pixels);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = y * width + x;
      if (!valid[i]) continue;
      let near = depth[i];
      let far = depth[i];
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          const ny = y + dy;
          if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
          const j = ny * width + nx;
          if (!valid[j]) continue;
          near = Math.min(near, depth[j]);
          far = Math.max(far, depth[j]);
        }
      }
      edge[i] = far > near * (1 + DEPTH_EDGE_TOLERANCE) ? 1 : 0;
    }
  }

  const joined = (a: number, b: number, c: number) => {
    if (!valid[a] || !valid[b] || !valid[c] || edge[a] || edge[b] || edge[c]) return false;
    // Also leaves out triangles seen almost edge-on from the camera, which are sheets stretched
    // across an outline rather than surfaces.
    const ax = positions[a * 3];
    const ay = positions[a * 3 + 1];
    const az = positions[a * 3 + 2];
    const ux = positions[b * 3] - ax;
    const uy = positions[b * 3 + 1] - ay;
    const uz = positions[b * 3 + 2] - az;
    const vx = positions[c * 3] - ax;
    const vy = positions[c * 3 + 1] - ay;
    const vz = positions[c * 3 + 2] - az;
    const fx = uy * vz - uz * vy;
    const fy = uz * vx - ux * vz;
    const fz = ux * vy - uy * vx;
    const facing = Math.abs(fx * ax + fy * ay + fz * az) / (Math.hypot(fx, fy, fz) * Math.hypot(ax, ay, az) || 1);
    return facing >= MIN_FACING;
  };

  const indices = new Uint32Array((width - 1) * (height - 1) * 6);
  let count = 0;
  for (let y = 0; y < height - 1; y++) {
    for (let x = 0; x < width - 1; x++) {
      const topLeft = y * width + x;
      const topRight = topLeft + 1;
      const bottomLeft = topLeft + width;
      const bottomRight = bottomLeft + 1;
      if (joined(topLeft, bottomLeft, topRight)) {
        indices[count++] = topLeft;
        indices[count++] = bottomLeft;
        indices[count++] = topRight;
      }
      if (joined(topRight, bottomLeft, bottomRight)) {
        indices[count++] = topRight;
        indices[count++] = bottomLeft;
        indices[count++] = bottomRight;
      }
    }
  }

  depths.sort((a, b) => a - b);
  const percentile = (p: number) => (depths.length ? depths[Math.floor((depths.length - 1) * p)] : 1);
  // MoGe's camera has a field of view of roughly 60 degrees when nothing can be fitted.
  const fallbackFocal = Math.max(width, height) / (2 * Math.tan(Math.PI / 6));

  return {
    positions,
    normals: meshNormals,
    uvs,
    indices: indices.slice(0, count),
    focusDistance: percentile(0.5),
    farDistance: percentile(0.99),
    focalLength: focalDenominator > 0 && focalNumerator > 0 ? focalNumerator / focalDenominator : fallbackFocal,
  };
}
