import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useTranslation } from 'react-i18next';
import { invoke } from '@tauri-apps/api/core';
import { Loader2, RotateCcw, X } from 'lucide-react';
import clsx from 'clsx';
import Text from '../ui/Text';
import { TextColors, TextVariants, TextWeights } from '../../types/typography';
import { useEditorStore } from '../../store/useEditorStore';
import { useProcessStore } from '../../store/useProcessStore';
import {
  colorizeDepth,
  drawOriented,
  normalMapImageData,
  orientationOf,
  parsePhotoGeometry,
  PhotoGeometry,
} from '../../utils/photoGeometry';
import type { SceneSurface } from './PhotoScene3D';

const PhotoScene3D = lazy(() => import('./PhotoScene3D'));

type ViewerTab = 'depth' | 'normals' | 'view3d';

interface MapState<T> {
  value?: T;
  error?: string;
  loading: boolean;
}

// The last photo's maps, so switching tabs or reopening the viewer doesn't calculate them again.
let cachedPath: string | null = null;
let cachedDepth: string | undefined;
let cachedGeometry: PhotoGeometry | undefined;
let lastTab: ViewerTab = 'depth';
let lastSurface: SceneSurface = 'photo';
let lastColorDepth = false;

interface GeometryViewerProps {
  isOpen: boolean;
  onClose(): void;
}

function SegmentedButtons<T extends string>({
  options,
  value,
  onChange,
}: {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange(value: T): void;
}) {
  return (
    <div className="flex bg-bg-primary rounded-lg p-0.5 gap-0.5">
      {options.map((option) => (
        <button
          key={option.value}
          className={clsx(
            'px-3 py-1 rounded-md text-sm transition-colors',
            option.value === value
              ? 'bg-accent text-button-text'
              : 'text-text-secondary hover:bg-card-active hover:text-text-primary',
          )}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

// Shows the photo's depth map, its normal map and its 3D shape, as the AI models see them.
export default function GeometryViewer({ isOpen, onClose }: GeometryViewerProps) {
  const { t } = useTranslation();
  const path = useEditorStore((s) => s.selectedImage?.path ?? null);
  const orientationSteps = useEditorStore((s) => s.adjustments.orientationSteps);
  const flipHorizontal = useEditorStore((s) => s.adjustments.flipHorizontal);
  const flipVertical = useEditorStore((s) => s.adjustments.flipVertical);
  const aiModelDownloadStatus = useProcessStore((s) => s.aiModelDownloadStatus);

  const [isMounted, setIsMounted] = useState(false);
  const [show, setShow] = useState(false);
  const [tab, setTab] = useState<ViewerTab>(lastTab);
  const [surface, setSurface] = useState<SceneSurface>(lastSurface);
  const [colorDepth, setColorDepth] = useState(lastColorDepth);
  const [resetCount, setResetCount] = useState(0);
  const [depth, setDepth] = useState<MapState<string>>({ loading: false });
  const [geometry, setGeometry] = useState<MapState<PhotoGeometry>>({ loading: false });
  const depthCanvasRef = useRef<HTMLCanvasElement>(null);
  const normalsCanvasRef = useRef<HTMLCanvasElement>(null);
  const openPathRef = useRef<string | null>(null);
  openPathRef.current = isOpen ? path : null;

  const orientation = useMemo(
    () => orientationOf({ orientationSteps, flipHorizontal, flipVertical }),
    [orientationSteps, flipHorizontal, flipVertical],
  );

  useEffect(() => {
    if (isOpen) {
      setIsMounted(true);
      const timer = setTimeout(() => setShow(true), 10);
      return () => clearTimeout(timer);
    }
    setShow(false);
    const timer = setTimeout(() => setIsMounted(false), 300);
    return () => clearTimeout(timer);
  }, [isOpen]);

  useEffect(() => {
    if (!isOpen) return;
    if (path !== cachedPath) {
      cachedPath = path;
      cachedDepth = undefined;
      cachedGeometry = undefined;
    }
    setDepth({ value: cachedDepth, loading: false });
    setGeometry({ value: cachedGeometry, loading: false });
  }, [isOpen, path]);

  useEffect(() => {
    if (!isOpen) return;
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const loadDepth = useCallback(() => {
    if (!path) return;
    setDepth({ loading: true });
    invoke<string>('generate_full_image_depth_map', { jsAdjustments: useEditorStore.getState().adjustments })
      .then((url) => {
        if (cachedPath === path) cachedDepth = url;
        if (openPathRef.current === path) setDepth({ value: url, loading: false });
      })
      .catch((e) => {
        if (openPathRef.current === path) setDepth({ error: String(e), loading: false });
      });
  }, [path]);

  const loadGeometry = useCallback(() => {
    if (!path) return;
    setGeometry({ loading: true });
    invoke<ArrayBuffer>('generate_photo_geometry', { jsAdjustments: useEditorStore.getState().adjustments })
      .then((buffer) => {
        const value = parsePhotoGeometry(buffer);
        if (cachedPath === path) cachedGeometry = value;
        if (openPathRef.current === path) setGeometry({ value, loading: false });
      })
      .catch((e) => {
        if (openPathRef.current === path) setGeometry({ error: String(e), loading: false });
      });
  }, [path]);

  const needed = tab === 'depth' ? depth : geometry;
  useEffect(() => {
    if (!isOpen || needed.value || needed.loading || needed.error) return;
    if (tab === 'depth') loadDepth();
    else loadGeometry();
  }, [isOpen, tab, needed, loadDepth, loadGeometry]);

  const normalMap = useMemo(
    () => (geometry.value ? normalMapImageData(geometry.value, orientation) : null),
    [geometry.value, orientation],
  );

  useEffect(() => {
    const canvas = depthCanvasRef.current;
    if (tab !== 'depth' || !canvas || !depth.value) return;
    const image = new Image();
    image.onload = () => {
      const { naturalWidth: width, naturalHeight: height } = image;
      let source: CanvasImageSource = image;
      if (colorDepth) {
        const scratch = document.createElement('canvas');
        scratch.width = width;
        scratch.height = height;
        const context = scratch.getContext('2d');
        if (context) {
          context.drawImage(image, 0, 0);
          context.putImageData(colorizeDepth(context.getImageData(0, 0, width, height)), 0, 0);
          source = scratch;
        }
      }
      drawOriented(canvas, source, width, height, orientation);
    };
    // The backend marks maps of the uncropped photo with a parameter images don't need.
    image.src = depth.value.replace(';space=source', '');
  }, [tab, depth.value, colorDepth, orientation, isMounted]);

  useEffect(() => {
    const canvas = normalsCanvasRef.current;
    if (tab !== 'normals' || !canvas || !normalMap) return;
    const scratch = document.createElement('canvas');
    scratch.width = normalMap.width;
    scratch.height = normalMap.height;
    scratch.getContext('2d')?.putImageData(normalMap, 0, 0);
    drawOriented(canvas, scratch, normalMap.width, normalMap.height, orientation);
  }, [tab, normalMap, orientation, isMounted]);

  const selectTab = (value: ViewerTab) => {
    lastTab = value;
    setTab(value);
  };
  const selectSurface = (value: SceneSurface) => {
    lastSurface = value;
    setSurface(value);
  };
  const selectColorDepth = (value: boolean) => {
    lastColorDepth = value;
    setColorDepth(value);
  };
  const retry = () => (tab === 'depth' ? setDepth({ loading: false }) : setGeometry({ loading: false }));

  if (!isMounted) return null;

  const status = (() => {
    if (needed.error) {
      return (
        <div className="flex flex-col items-center gap-3 max-w-md text-center">
          <Text color={TextColors.error}>{t('modals.geometry.failed')}</Text>
          <Text variant={TextVariants.small}>{needed.error}</Text>
          <button
            className="px-3 py-1 rounded-md bg-card-active text-text-primary hover:bg-accent hover:text-button-text"
            onClick={retry}
          >
            {t('modals.geometry.retry')}
          </button>
        </div>
      );
    }
    if (!needed.value) {
      return (
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="animate-spin text-text-secondary" size={28} />
          <Text variant={TextVariants.small}>
            {aiModelDownloadStatus
              ? t('modals.geometry.downloading', { status: aiModelDownloadStatus })
              : t('modals.geometry.calculating')}
          </Text>
        </div>
      );
    }
    return null;
  })();

  const caption =
    tab === 'depth'
      ? t('modals.geometry.depthCaption')
      : tab === 'normals'
        ? t('modals.geometry.normalsCaption')
        : t('modals.geometry.view3dCaption');

  return createPortal(
    <div
      className={clsx(
        'fixed inset-0 flex items-center justify-center z-50 bg-black/40 backdrop-blur-xs transition-opacity duration-300 ease-in-out',
        show ? 'opacity-100' : 'opacity-0',
      )}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div
        className={clsx(
          'bg-surface rounded-xl shadow-2xl p-4 flex flex-col gap-3 w-[min(1400px,94vw)] h-[88vh] transform transition-all duration-300 ease-out',
          show ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 -translate-y-4',
        )}
      >
        <div className="flex items-center justify-between gap-4">
          <Text variant={TextVariants.title} weight={TextWeights.semibold}>
            {t('modals.geometry.title')}
          </Text>
          <SegmentedButtons
            options={[
              { value: 'depth', label: t('modals.geometry.tabDepth') },
              { value: 'normals', label: t('modals.geometry.tabNormals') },
              { value: 'view3d', label: t('modals.geometry.tab3d') },
            ]}
            value={tab}
            onChange={selectTab}
          />
          <button
            className="p-1.5 rounded-full text-text-secondary hover:bg-card-active hover:text-text-primary"
            onClick={onClose}
            data-tooltip={t('modals.geometry.close')}
          >
            <X size={20} />
          </button>
        </div>

        <div className="flex-1 min-h-0 relative rounded-lg overflow-hidden bg-[#111] flex items-center justify-center">
          {status ??
            (tab === 'depth' ? (
              <canvas ref={depthCanvasRef} className="w-full h-full object-contain" />
            ) : tab === 'normals' ? (
              <canvas ref={normalsCanvasRef} className="w-full h-full object-contain" />
            ) : (
              geometry.value &&
              normalMap && (
                <Suspense fallback={<Loader2 className="animate-spin text-text-secondary" size={28} />}>
                  <PhotoScene3D
                    geometry={geometry.value}
                    normalMap={normalMap}
                    orientation={orientation}
                    surface={surface}
                    resetCount={resetCount}
                  />
                </Suspense>
              )
            ))}
        </div>

        <div className="flex items-center justify-between gap-4 min-h-8">
          <Text variant={TextVariants.small}>{caption}</Text>
          {tab === 'depth' && (
            <SegmentedButtons
              options={[
                { value: 'gray', label: t('modals.geometry.gray') },
                { value: 'color', label: t('modals.geometry.color') },
              ]}
              value={colorDepth ? 'color' : 'gray'}
              onChange={(value) => selectColorDepth(value === 'color')}
            />
          )}
          {tab === 'view3d' && (
            <div className="flex items-center gap-2 shrink-0">
              <SegmentedButtons
                options={[
                  { value: 'photo', label: t('modals.geometry.surfacePhoto') },
                  { value: 'normals', label: t('modals.geometry.surfaceNormals') },
                  { value: 'clay', label: t('modals.geometry.surfaceClay') },
                ]}
                value={surface}
                onChange={selectSurface}
              />
              <button
                className="flex items-center gap-1.5 px-3 py-1 rounded-md text-sm bg-bg-primary text-text-secondary hover:bg-card-active hover:text-text-primary"
                onClick={() => setResetCount((count) => count + 1)}
              >
                <RotateCcw size={14} />
                {t('modals.geometry.resetView')}
              </button>
            </div>
          )}
        </div>
      </div>
    </div>,
    document.body,
  );
}
