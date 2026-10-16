import { useEffect, useMemo } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { useTranslation } from 'react-i18next';
import { toast } from 'react-toastify';
import { useLibraryStore } from '../store/useLibraryStore';
import { Invokes } from '../components/ui/AppProperties';

interface SemanticMatch {
  path: string;
  score: number;
}

/**
 * Runs the natural-language search in `searchCriteria.semantic` against the current image list,
 * again whenever the list's paths change. The first search in a folder analyses its photos,
 * which `semanticResults.progress` reports.
 */
export function useSemanticSearch() {
  const { t } = useTranslation();
  const query = useLibraryStore((state) => state.searchCriteria.semantic);
  const imageList = useLibraryStore((state) => state.imageList);
  const setLibrary = useLibraryStore((state) => state.setLibrary);
  const pathsKey = useMemo(() => imageList.map((image) => image.path).join('\n'), [imageList]);

  useEffect(() => {
    if (!query) {
      setLibrary({ semanticResults: null });
      return;
    }

    let isActive = true;
    const paths = pathsKey ? pathsKey.split('\n') : [];
    setLibrary((state) => ({
      semanticResults: {
        query,
        scores: state.semanticResults?.query === query ? state.semanticResults.scores : null,
        progress: null,
        isSearching: true,
      },
    }));

    const unlisten = listen<{ current: number; total: number }>('semantic-search-progress', (event) => {
      if (!isActive) return;
      setLibrary((state) => ({
        semanticResults: state.semanticResults && { ...state.semanticResults, progress: event.payload },
      }));
    });

    invoke<SemanticMatch[]>(Invokes.SemanticSearch, { query, paths })
      .then((matches) => {
        if (!isActive) return;
        setLibrary({
          semanticResults: {
            query,
            scores: Object.fromEntries(matches.map((match) => [match.path, match.score])),
            progress: null,
            isSearching: false,
          },
        });
      })
      .catch((err) => {
        if (!isActive) return;
        console.error('AI search failed:', err);
        toast.error(t('library.header.search.aiSearchFailed', { error: String(err) }));
        setLibrary({ semanticResults: { query, scores: null, progress: null, isSearching: false } });
      });

    return () => {
      isActive = false;
      unlisten.then((stop) => stop());
    };
  }, [query, pathsKey, setLibrary, t]);
}
