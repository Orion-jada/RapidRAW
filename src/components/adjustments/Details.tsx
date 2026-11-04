import { memo, useEffect, useState } from 'react';
import { useTranslation } from 'react-i18next';
import { Loader2 } from 'lucide-react';
import { invoke } from '@tauri-apps/api/core';
import Dropdown from '../ui/Dropdown';
import Slider, { SliderChangeEvent } from '../ui/Slider';
import Text from '../ui/Text';
import { TextVariants } from '../../types/typography';
import { SpatialEffect } from './Effects';
import {
  Adjustments,
  DenoiseMethod,
  DetailsAdjustment,
  getAdjustmentToolOrder,
  getAiDenoiseEnablePatch,
  getDenoiseModelName,
  getDenoiseModelOptions,
  getHiddenAdjustmentTools,
  resolveDenoiseModel,
} from '../../utils/adjustments';
import { AppSettings, Invokes } from '../ui/AppProperties';
import AdjustmentSubSection from './AdjustmentSubSection';
import { canvasToolState, useEditorStore } from '../../store/useEditorStore';
import { selectAiModelDownloadStatus, useProcessStore } from '../../store/useProcessStore';
import { areSectionPropsEqual } from '../../utils/adjustments';

interface DetailsPanelProps {
  adjustments: Adjustments;
  setAdjustments(adjustments: Partial<Adjustments>): any;
  appSettings: AppSettings | null;
  isForMask?: boolean;
  onDragStateChange?: (isDragging: boolean) => void;
}

function DetailsPanel({
  adjustments,
  setAdjustments,
  appSettings,
  isForMask = false,
  onDragStateChange,
}: DetailsPanelProps) {
  const { t } = useTranslation();
  const setEditor = useEditorStore((state) => state.setEditor);
  const aiDenoiseProgress = useEditorStore((state) =>
    state.selectedImage ? state.aiDenoiseProgress[state.selectedImage.path] : undefined,
  );
  const aiModelDownloadStatus = useProcessStore(selectAiModelDownloadStatus);
  const selectedPath = useEditorStore((state) => state.selectedImage?.path);
  const isRaw = useEditorStore((state) => !!state.selectedImage?.isRaw);
  const [raw9Available, setRaw9Available] = useState(false);

  const handleAdjustmentChange = (key: string, value: string) => {
    const numericValue = parseInt(value, 10);
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: numericValue }));
  };

  const handleFloatAdjustmentChange = (key: string, value: string) => {
    setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, [key]: parseFloat(value) }));
  };

  const isAiFree = appSettings?.aiProvider === 'ai-free';
  const canPickDenoiseModel = !isForMask && !isAiFree && !!adjustments.aiDenoiseEnabled;

  useEffect(() => {
    if (!canPickDenoiseModel || !isRaw || !selectedPath) {
      setRaw9Available(false);
      return;
    }
    let cancelled = false;
    invoke<boolean>(Invokes.IsRaw9Available, { paths: [selectedPath] })
      .then((available) => {
        if (!cancelled) setRaw9Available(available);
      })
      .catch(() => {
        if (!cancelled) setRaw9Available(false);
      });
    return () => {
      cancelled = true;
    };
  }, [canPickDenoiseModel, isRaw, selectedPath]);

  const denoiseMethod = adjustments.aiDenoiseMethod ?? DenoiseMethod.Ai;
  const defaultDenoiseModel = resolveDenoiseModel(
    DenoiseMethod.Ai,
    appSettings?.liveDenoiseModel,
    isRaw,
    raw9Available,
  );
  const denoiseModel = resolveDenoiseModel(denoiseMethod, appSettings?.liveDenoiseModel, isRaw, raw9Available);
  const denoiseModelOptions = [
    {
      label: t('adjustments.details.aiDenoiseModelDefault', { model: getDenoiseModelName(defaultDenoiseModel) }),
      value: DenoiseMethod.Ai,
    },
    ...getDenoiseModelOptions(t, isRaw, raw9Available),
  ];
  const hiddenTools = getHiddenAdjustmentTools(appSettings?.adjustmentLayout);
  const toolOrder = getAdjustmentToolOrder('details', appSettings?.adjustmentLayout?.toolOrder);

  return (
    <div className="flex flex-col gap-4">
      {!hiddenTools.includes('sharpening') && (
        <AdjustmentSubSection
          id="sharpening"
          order={toolOrder.indexOf('sharpening')}
          title={t('adjustments.details.sharpening')}
        >
          <Slider
            label={t('adjustments.details.sharpness')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Sharpness, e.target.value)}
            step={1}
            value={adjustments.sharpness}
            onDragStateChange={onDragStateChange}
          />
          {!isForMask && (
            <Slider
              label={t('adjustments.details.threshold')}
              max={80}
              min={0}
              onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.SharpnessThreshold, e.target.value)}
              step={1}
              value={adjustments.sharpnessThreshold ?? 15}
              onDragStateChange={onDragStateChange}
              defaultValue={15}
              fillOrigin="min"
            />
          )}
        </AdjustmentSubSection>
      )}

      {!hiddenTools.includes('presence') && (
        <AdjustmentSubSection
          id="presence"
          order={toolOrder.indexOf('presence')}
          title={t('adjustments.details.presence')}
        >
          <Slider
            label={t('adjustments.details.clarity')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Clarity, e.target.value)}
            speedEditId={DetailsAdjustment.Clarity}
            step={1}
            value={adjustments.clarity}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.dehaze')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Dehaze, e.target.value)}
            speedEditId={DetailsAdjustment.Dehaze}
            step={1}
            value={adjustments.dehaze}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.structure')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Structure, e.target.value)}
            speedEditId={DetailsAdjustment.Structure}
            step={1}
            value={adjustments.structure}
            onDragStateChange={onDragStateChange}
          />
          {!isForMask && (
            <div
              data-center-marker-controls="centre"
              onPointerDown={() => setEditor(canvasToolState({ activeCenterMarker: 'centre' }))}
            >
              <Slider
                label={t('adjustments.details.centre')}
                max={100}
                min={-100}
                onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.Centré, e.target.value)}
                step={1}
                value={adjustments.centré}
                onDragStateChange={onDragStateChange}
              />
            </div>
          )}
        </AdjustmentSubSection>
      )}

      {!hiddenTools.includes('noiseReduction') && (
        <AdjustmentSubSection
          id="noiseReduction"
          order={toolOrder.indexOf('noiseReduction')}
          title={t('adjustments.details.noiseReduction')}
        >
          <Slider
            label={t('adjustments.details.luminance')}
            max={100}
            min={isForMask ? -100 : 0}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.LumaNoiseReduction, e.target.value)}
            step={1}
            value={adjustments.lumaNoiseReduction}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.color')}
            max={100}
            min={isForMask ? -100 : 0}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.ColorNoiseReduction, e.target.value)}
            step={1}
            value={adjustments.colorNoiseReduction}
            onDragStateChange={onDragStateChange}
          />
          {!isForMask && (
            <div className="mt-2">
              <SpatialEffect
                enabled={!!adjustments.aiDenoiseEnabled}
                label={
                  (isAiFree && !adjustments.aiDenoiseEnabled) || denoiseModel === DenoiseMethod.Bm3d
                    ? t('adjustments.details.classicDenoise')
                    : t('adjustments.details.aiDenoise')
                }
                onToggle={(enabled: boolean) =>
                  setAdjustments((prev: Partial<Adjustments>) => ({
                    ...prev,
                    ...(enabled ? getAiDenoiseEnablePatch(appSettings?.aiProvider) : { aiDenoiseEnabled: false }),
                  }))
                }
              >
                {canPickDenoiseModel && (
                  <div>
                    <Text variant={TextVariants.label} className="mb-2 block">
                      {t('adjustments.details.aiDenoiseModel')}
                    </Text>
                    <Dropdown<DenoiseMethod>
                      options={denoiseModelOptions}
                      value={denoiseMethod === DenoiseMethod.Ai ? DenoiseMethod.Ai : denoiseModel}
                      onChange={(method) =>
                        setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, aiDenoiseMethod: method }))
                      }
                      disabled={aiDenoiseProgress !== undefined}
                    />
                  </div>
                )}
                {aiDenoiseProgress !== undefined ? (
                  <div className="flex flex-col items-center justify-center gap-1 p-4 text-text-secondary text-center">
                    <div className="flex items-center gap-2">
                      <Loader2 size={16} className="animate-spin shrink-0" />
                      <Text variant={TextVariants.label}>
                        {aiModelDownloadStatus
                          ? t('editor.masks.settings.aiModelDownloading')
                          : t('editor.ai.denoising')}
                      </Text>
                    </div>
                    <Text variant={TextVariants.small} className="text-accent">
                      {aiModelDownloadStatus || `${Math.round(aiDenoiseProgress * 100)}%`}
                    </Text>
                  </div>
                ) : (
                  <>
                    <Slider
                      label={t('adjustments.effects.amount')}
                      max={100}
                      min={0}
                      defaultValue={100}
                      onChange={(e: SliderChangeEvent) =>
                        handleAdjustmentChange(DetailsAdjustment.AiDenoiseAmount, String(e.target.value))
                      }
                      step={1}
                      value={adjustments.aiDenoiseAmount ?? 100}
                      onDragStateChange={onDragStateChange}
                      fillOrigin="min"
                    />
                    <Slider
                      label={t('adjustments.details.aiDenoiseDetail')}
                      max={100}
                      min={0}
                      onChange={(e: SliderChangeEvent) =>
                        handleAdjustmentChange(DetailsAdjustment.AiDenoiseDetail, String(e.target.value))
                      }
                      step={1}
                      value={adjustments.aiDenoiseDetail ?? 0}
                      onDragStateChange={onDragStateChange}
                    />
                  </>
                )}
              </SpatialEffect>
            </div>
          )}
        </AdjustmentSubSection>
      )}

      {!isForMask && !hiddenTools.includes('chromaticAberration') && (
        <AdjustmentSubSection
          id="chromaticAberration"
          order={toolOrder.indexOf('chromaticAberration')}
          title={t('adjustments.details.chromaticAberration')}
        >
          <Slider
            label={t('adjustments.details.redCyan')}
            max={100}
            min={-100}
            onChange={(e: any) => handleAdjustmentChange(DetailsAdjustment.ChromaticAberrationRedCyan, e.target.value)}
            step={1}
            value={adjustments.chromaticAberrationRedCyan}
            onDragStateChange={onDragStateChange}
          />
          <Slider
            label={t('adjustments.details.blueYellow')}
            max={100}
            min={-100}
            onChange={(e: any) =>
              handleAdjustmentChange(DetailsAdjustment.ChromaticAberrationBlueYellow, e.target.value)
            }
            step={1}
            value={adjustments.chromaticAberrationBlueYellow}
            onDragStateChange={onDragStateChange}
          />
        </AdjustmentSubSection>
      )}

      {!isForMask && !hiddenTools.includes('negativeConversion') && (
        <AdjustmentSubSection
          id="negativeConversion"
          order={toolOrder.indexOf('negativeConversion')}
          title={t('adjustments.details.negativeConversion')}
        >
          <SpatialEffect
            enabled={!!adjustments.negativeConversionEnabled}
            label={t('adjustments.details.convertNegative')}
            onToggle={(enabled: boolean) =>
              setAdjustments((prev: Partial<Adjustments>) => ({ ...prev, negativeConversionEnabled: enabled }))
            }
          >
            <Slider
              label={t('adjustments.details.negativeRed')}
              max={2}
              min={0.5}
              defaultValue={1}
              onChange={(e: SliderChangeEvent) =>
                handleFloatAdjustmentChange(DetailsAdjustment.NegativeRedWeight, String(e.target.value))
              }
              step={0.01}
              value={adjustments.negativeRedWeight ?? 1}
              onDragStateChange={onDragStateChange}
              fillOrigin="min"
            />
            <Slider
              label={t('adjustments.details.negativeGreen')}
              max={2}
              min={0.5}
              defaultValue={1}
              onChange={(e: SliderChangeEvent) =>
                handleFloatAdjustmentChange(DetailsAdjustment.NegativeGreenWeight, String(e.target.value))
              }
              step={0.01}
              value={adjustments.negativeGreenWeight ?? 1}
              onDragStateChange={onDragStateChange}
              fillOrigin="min"
            />
            <Slider
              label={t('adjustments.details.negativeBlue')}
              max={2}
              min={0.5}
              defaultValue={1}
              onChange={(e: SliderChangeEvent) =>
                handleFloatAdjustmentChange(DetailsAdjustment.NegativeBlueWeight, String(e.target.value))
              }
              step={0.01}
              value={adjustments.negativeBlueWeight ?? 1}
              onDragStateChange={onDragStateChange}
              fillOrigin="min"
            />
            <Slider
              label={t('adjustments.details.negativeExposure')}
              max={2}
              min={-2}
              onChange={(e: SliderChangeEvent) =>
                handleFloatAdjustmentChange(DetailsAdjustment.NegativeExposure, String(e.target.value))
              }
              step={0.05}
              value={adjustments.negativeExposure ?? 0}
              onDragStateChange={onDragStateChange}
            />
            <Slider
              label={t('adjustments.details.negativeContrast')}
              max={2.5}
              min={0.5}
              defaultValue={1}
              onChange={(e: SliderChangeEvent) =>
                handleFloatAdjustmentChange(DetailsAdjustment.NegativeContrast, String(e.target.value))
              }
              step={0.05}
              value={adjustments.negativeContrast ?? 1}
              onDragStateChange={onDragStateChange}
              fillOrigin="min"
            />
          </SpatialEffect>
        </AdjustmentSubSection>
      )}
    </div>
  );
}

export default memo(DetailsPanel, areSectionPropsEqual('details'));
