import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { buildPhotoMesh, MapOrientation, PhotoGeometry } from '../../utils/photoGeometry';

export type SceneSurface = 'photo' | 'normals' | 'clay';

const MAX_ORBIT = (70 * Math.PI) / 180;

interface PhotoScene3DProps {
  geometry: PhotoGeometry;
  normalMap: ImageData;
  orientation: MapOrientation;
  surface: SceneSurface;
  // Changing it puts the camera back where the photo was taken from.
  resetCount: number;
}

interface SceneState {
  camera: THREE.PerspectiveCamera;
  controls: OrbitControls;
  mesh: THREE.Mesh;
  materials: Record<SceneSurface, THREE.Material>;
  reset(): void;
}

// The photo's geometry as a mesh you can orbit, starting from the photo's own viewpoint.
export default function PhotoScene3D({ geometry, normalMap, orientation, surface, resetCount }: PhotoScene3DProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<SceneState | null>(null);
  const surfaceRef = useRef(surface);
  surfaceRef.current = surface;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    const mesh = buildPhotoMesh(geometry);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x111111);

    const camera = new THREE.PerspectiveCamera(50, 1, mesh.focusDistance * 0.01, mesh.farDistance * 20);
    scene.add(camera);
    scene.add(new THREE.AmbientLight(0xffffff, 0.45));
    // Lights the clay surface from above the viewer's right shoulder, wherever the viewer goes.
    const light = new THREE.DirectionalLight(0xffffff, 2.2);
    light.position.set(1, 1.5, 0.5);
    light.target.position.set(0, 0, -1);
    camera.add(light, light.target);

    const bufferGeometry = new THREE.BufferGeometry();
    bufferGeometry.setAttribute('position', new THREE.BufferAttribute(mesh.positions, 3));
    bufferGeometry.setAttribute('normal', new THREE.BufferAttribute(mesh.normals, 3));
    bufferGeometry.setAttribute('uv', new THREE.BufferAttribute(mesh.uvs, 2));
    bufferGeometry.setIndex(new THREE.BufferAttribute(mesh.indices, 1));

    const makeTexture = (data: Uint8Array | Uint8ClampedArray, width: number, height: number) => {
      const texture = new THREE.DataTexture(data, width, height, THREE.RGBAFormat);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.generateMipmaps = true;
      texture.minFilter = THREE.LinearMipmapLinearFilter;
      texture.magFilter = THREE.LinearFilter;
      texture.anisotropy = renderer.capabilities.getMaxAnisotropy();
      texture.needsUpdate = true;
      return texture;
    };
    const photoTexture = makeTexture(geometry.texture.data, geometry.texture.width, geometry.texture.height);
    const normalTexture = makeTexture(normalMap.data, normalMap.width, normalMap.height);
    const materials: Record<SceneSurface, THREE.Material> = {
      photo: new THREE.MeshBasicMaterial({ map: photoTexture, side: THREE.DoubleSide }),
      normals: new THREE.MeshBasicMaterial({ map: normalTexture, side: THREE.DoubleSide }),
      clay: new THREE.MeshStandardMaterial({ color: 0xd9d6d0, roughness: 0.85, side: THREE.DoubleSide }),
    };

    // Turned and flipped like the photo in the editor: the turn is a roll around the camera's
    // view, and the flip is applied after it.
    const photoMesh = new THREE.Mesh(bufferGeometry, materials[surfaceRef.current]);
    photoMesh.rotation.z = (-orientation.steps * Math.PI) / 2;
    const flipped = new THREE.Group();
    flipped.scale.set(orientation.flipHorizontal ? -1 : 1, orientation.flipVertical ? -1 : 1, 1);
    flipped.add(photoMesh);
    scene.add(flipped);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.12;
    controls.rotateSpeed = 0.35;
    // Beyond about 70 degrees from where the photo was taken there's little but its back to see.
    controls.minAzimuthAngle = -MAX_ORBIT;
    controls.maxAzimuthAngle = MAX_ORBIT;
    controls.minPolarAngle = Math.PI / 2 - MAX_ORBIT;
    controls.maxPolarAngle = Math.PI / 2 + MAX_ORBIT;
    controls.minDistance = mesh.focusDistance * 0.05;
    controls.maxDistance = mesh.focusDistance * 10;
    controls.zoomToCursor = true;

    const sideways = orientation.steps % 2 === 1;
    const photoWidth = sideways ? geometry.height : geometry.width;
    const photoHeight = sideways ? geometry.width : geometry.height;
    const fitView = () => {
      const width = container.clientWidth;
      const height = container.clientHeight;
      if (width === 0 || height === 0) return;
      renderer.setSize(width, height);
      camera.aspect = width / height;
      // The vertical field of view that shows the whole photo from where it was taken.
      const halfHeight = Math.max(photoHeight / 2, photoWidth / 2 / camera.aspect) / mesh.focalLength;
      camera.fov = (2 * Math.atan(halfHeight * 1.04) * 180) / Math.PI;
      camera.updateProjectionMatrix();
    };

    const reset = () => {
      camera.position.set(0, 0, 0);
      controls.target.set(0, 0, -mesh.focusDistance);
      controls.update();
    };
    reset();
    fitView();

    const resizeObserver = new ResizeObserver(fitView);
    resizeObserver.observe(container);
    renderer.setAnimationLoop(() => {
      controls.update();
      renderer.render(scene, camera);
    });

    sceneRef.current = { camera, controls, mesh: photoMesh, materials, reset };

    return () => {
      sceneRef.current = null;
      resizeObserver.disconnect();
      renderer.setAnimationLoop(null);
      controls.dispose();
      bufferGeometry.dispose();
      Object.values(materials).forEach((material) => material.dispose());
      photoTexture.dispose();
      normalTexture.dispose();
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [geometry, normalMap, orientation]);

  useEffect(() => {
    const state = sceneRef.current;
    if (state) state.mesh.material = state.materials[surface];
  }, [surface]);

  useEffect(() => {
    sceneRef.current?.reset();
  }, [resetCount]);

  return <div ref={containerRef} className="absolute inset-0 cursor-grab active:cursor-grabbing" />;
}
