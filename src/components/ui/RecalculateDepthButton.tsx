import clsx from 'clsx';
import { Loader2, RefreshCw } from 'lucide-react';
import { useTranslation } from 'react-i18next';

interface RecalculateDepthButtonProps {
  onClick(): void;
  isCalculating?: boolean;
}

/** Calculates a depth map again, e.g. one saved by an older depth model. */
export default function RecalculateDepthButton({ onClick, isCalculating = false }: RecalculateDepthButtonProps) {
  const { t } = useTranslation();

  return (
    <button
      type="button"
      onClick={onClick}
      disabled={isCalculating}
      data-tooltip={t('editor.ai.recalculateDepthTooltip')}
      className={clsx(
        'flex w-full items-center justify-center gap-2 p-2 rounded-md text-sm font-medium text-text-secondary',
        'transition-colors bg-surface hover:bg-card-active hover:text-text-primary',
        'disabled:opacity-50 disabled:cursor-not-allowed disabled:hover:bg-surface',
      )}
    >
      {isCalculating ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
      <span className="select-none">{t('editor.ai.recalculateDepth')}</span>
    </button>
  );
}
