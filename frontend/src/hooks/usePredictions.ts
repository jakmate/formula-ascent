import { useState, useEffect, useCallback, useRef } from 'react';
import type { SystemStatus } from '../types/SystemStatus';
import type { ModelResults } from '../types/ModelResults';

export type SeriesType = 'f3_to_f2' | 'f2_to_f1';

// Parse fragment: #f3_to_f2/LightGBM
const parseFragment = (): { series: SeriesType; model: string | null } => {
  const hash = globalThis.location.hash.slice(1);
  const [series, encodedModel] = hash.split('/');
  const validSeries: SeriesType = ['f3_to_f2', 'f2_to_f1'].includes(series)
    ? (series as SeriesType)
    : 'f3_to_f2';
  const model = encodedModel ? decodeURIComponent(encodedModel) : null;
  return { series: validSeries, model };
};

// Set fragment: #series/model
const setFragment = (series: SeriesType, model: string) => {
  globalThis.location.hash = `${series}/${encodeURIComponent(model)}`;
};

export const usePredictions = (initialSeries: SeriesType = 'f3_to_f2') => {
  const { series: fragmentSeries, model: fragmentModel } = parseFragment();
  const [series, setSeries] = useState<SeriesType>(
    fragmentSeries || initialSeries
  );
  const [modelsList, setModelsList] = useState<string[]>([]);
  const [selectedModel, setSelectedModel] = useState<string>(
    fragmentModel || ''
  );
  const [currentModelResults, setCurrentModelResults] =
    useState<ModelResults | null>(null);
  const [systemStatus, setSystemStatus] = useState<SystemStatus | null>(null);
  const [loadingModels, setLoadingModels] = useState<boolean>(true);
  const [loadingPredictions, setLoadingPredictions] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const refreshStatusRef = useRef<SystemStatus | null>(null);
  const lastFetchedKeyRef = useRef<string>('');

  const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const fetchModelsAndStatus = useCallback(
    async (targetSeries: SeriesType) => {
      try {
        setLoadingModels(true);
        const response = await fetch(`${API_BASE}/api/models/${targetSeries}`);
        if (!response.ok) throw new Error('Failed to fetch models');
        const data = await response.json();
        setModelsList(data.models || []);
        setSystemStatus(data.system_status);
        return data.models as string[];
      } catch (err) {
        setError('Failed to load models list');
        console.error(err);
        return [];
      } finally {
        setLoadingModels(false);
      }
    },
    [API_BASE]
  );

  const fetchModelPredictions = useCallback(
    async (targetSeries: SeriesType, modelName: string, force = false) => {
      if (!modelName) return null;
      const key = `${targetSeries}/${modelName}`;
      if (!force && lastFetchedKeyRef.current === key) return null;
      try {
        setLoadingPredictions(true);
        const response = await fetch(
          `${API_BASE}/api/predictions/${targetSeries}/${encodeURIComponent(modelName)}`
        );
        if (!response.ok) {
          if (response.status === 404) throw new Error('Model not found');
          throw new Error('Server error');
        }
        const data: ModelResults = await response.json();
        setCurrentModelResults(data);
        lastFetchedKeyRef.current = key;
        setError(null);
        return data;
      } catch (err) {
        setError(
          err instanceof Error ? err.message : 'Failed to load predictions'
        );
        console.error(err);
        return null;
      } finally {
        setLoadingPredictions(false);
      }
    },
    [API_BASE]
  );

  const selectedModelRef = useRef(selectedModel);
  useEffect(() => {
    selectedModelRef.current = selectedModel;
  }, [selectedModel]);

  // Load models on series change
  useEffect(() => {
    const loadModels = async () => {
      const models = await fetchModelsAndStatus(series);
      const currentSelected = selectedModelRef.current;
      let targetModel = currentSelected;
      if (!targetModel || !models.includes(targetModel)) {
        targetModel =
          fragmentModel && models.includes(fragmentModel)
            ? fragmentModel
            : models[0] || '';
      }
      if (targetModel && targetModel !== currentSelected) {
        setSelectedModel(targetModel);
      }
    };
    loadModels();
  }, [series, fragmentModel, fetchModelsAndStatus]);

  // Fetch predictions when selectedModel or modelsList changes
  useEffect(() => {
    if (selectedModel && modelsList.includes(selectedModel)) {
      const run = async () => {
        await fetchModelPredictions(series, selectedModel);
      };
      run();
    }
  }, [selectedModel, modelsList, series, fetchModelPredictions]);

  const refreshPredictions = async () => {
    try {
      setLoadingModels(true);
      setError(null);
      refreshStatusRef.current = systemStatus;

      const refreshResponse = await fetch(
        `${API_BASE}/api/system/refresh/predictions`,
        { method: 'POST' }
      );
      if (!refreshResponse.ok) throw new Error('Refresh request failed');

      const maxAttempts = 10;
      let attempts = 0;

      const checkForUpdates = async () => {
        try {
          const modelsResponse = await fetch(
            `${API_BASE}/api/models/${series}`
          );
          if (!modelsResponse.ok)
            throw new Error('Failed to fetch updated status');
          const data = await modelsResponse.json();
          setModelsList(data.models);
          setSystemStatus(data.system_status);

          const hasNewData =
            data.system_status?.last_scrape_full !==
              refreshStatusRef.current?.last_scrape_full ||
            data.system_status?.last_scrape_predictions !==
              refreshStatusRef.current?.last_scrape_predictions;

          if (hasNewData && selectedModel) {
            await fetchModelPredictions(series, selectedModel, true);
            setLoadingModels(false);
            return;
          }

          if (++attempts >= maxAttempts) {
            setLoadingModels(false);
            setError('Update check timeout');
            return;
          }

          setTimeout(checkForUpdates, 3000);
        } catch (err) {
          setLoadingModels(false);
          setError(
            `Update failed: ${err instanceof Error ? err.message : 'Unknown error'}`
          );
        }
      };

      await checkForUpdates();
    } catch (err) {
      setLoadingModels(false);
      setError('Could not refresh data. Please try again later.');
      console.error('Refresh error:', err);
    }
  };

  useEffect(() => {
    if (selectedModel) {
      setFragment(series, selectedModel);
    }
  }, [series, selectedModel]);

  useEffect(() => {
    const handleHashChange = () => {
      const { series: newSeries, model: newModel } = parseFragment();
      if (newSeries !== series) setSeries(newSeries);
      if (newModel && newModel !== selectedModel) setSelectedModel(newModel);
    };
    globalThis.addEventListener('hashchange', handleHashChange);
    return () => globalThis.removeEventListener('hashchange', handleHashChange);
  }, [series, selectedModel]);

  return {
    predictions: currentModelResults?.predictions || [],
    selectedModel,
    setSelectedModel,
    models: modelsList,
    loading: loadingModels || loadingPredictions,
    status: systemStatus,
    error,
    refreshPredictions,
    currentPredictions: currentModelResults?.predictions || [],
    series,
    setSeries,
  };
};
