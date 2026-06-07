import { renderHook, waitFor, act } from '@testing-library/react';
import { describe, it, expect, vi, beforeEach } from 'vitest';
import { usePredictions } from './usePredictions';

const mockFetch = vi.fn();
globalThis.fetch = mockFetch;
vi.spyOn(console, 'error').mockImplementation(() => {});

const mockModelsResponse = {
  models: ['model1', 'model2'],
  system_status: {
    last_scrape_predictions: '2023-01-01T00:00:00Z',
    last_scrape_full: '2023-01-01T00:00:00Z',
    last_training: '2023-01-02T00:00:00Z',
  },
};

const mockPredictionsResponse = {
  model_name: 'model1',
  predictions: [{ id: 1 }],
  accuracy_metrics: { total_predictions: 1 },
};

// Helper: mock initial load (models fetch + predictions fetch)
const mockInitialLoad = (
  modelsResp = mockModelsResponse,
  predictionsResp = mockPredictionsResponse
) => {
  mockFetch
    .mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(modelsResp),
    })
    .mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(predictionsResp),
    });
};

describe('usePredictions', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    globalThis.location.hash = '';
  });

  it('loads models then predictions on mount', async () => {
    mockInitialLoad();

    const { result } = renderHook(() => usePredictions('f3_to_f2'));

    expect(result.current.loading).toBe(true);

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.models).toEqual(['model1', 'model2']);
    expect(result.current.selectedModel).toBe('model1');
    expect(result.current.status).toEqual(mockModelsResponse.system_status);
    expect(result.current.predictions).toEqual([{ id: 1 }]);

    expect(mockFetch).toHaveBeenNthCalledWith(
      1,
      'http://localhost:8000/api/models/f3_to_f2'
    );
    expect(mockFetch).toHaveBeenNthCalledWith(
      2,
      'http://localhost:8000/api/predictions/f3_to_f2/model1'
    );
  });

  it('handles models fetch failure', async () => {
    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toBe('Failed to load models list');
    expect(result.current.models).toEqual([]);
  });

  it('handles network error on models fetch', async () => {
    mockFetch.mockRejectedValueOnce(new Error('Network error'));

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.error).toBe('Failed to load models list');
  });

  it('updates selected model and fetches its predictions', async () => {
    mockInitialLoad();

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    const model2Response = {
      ...mockPredictionsResponse,
      model_name: 'model2',
      predictions: [{ id: 2 }],
    };
    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve(model2Response),
    });

    act(() => result.current.setSelectedModel('model2'));

    await waitFor(() =>
      expect(result.current.predictions).toEqual([{ id: 2 }])
    );

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/predictions/f3_to_f2/model2'
    );
  });

  it('does not refetch predictions for same series/model key', async () => {
    mockInitialLoad();

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    const callCount = mockFetch.mock.calls.length; // 2

    // Force same model selection — should not trigger new fetch
    act(() => result.current.setSelectedModel('model1'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(mockFetch).toHaveBeenCalledTimes(callCount);
  });

  it('fetches new predictions on series change', async () => {
    mockInitialLoad();

    const { result } = renderHook(() => usePredictions('f3_to_f2'));

    await waitFor(() => expect(result.current.loading).toBe(false));

    // Series switch: new models fetch + new predictions fetch
    mockFetch
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockModelsResponse),
      })
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockPredictionsResponse,
            predictions: [{ id: 99 }],
          }),
      });

    act(() => result.current.setSeries('f2_to_f1'));

    await waitFor(() =>
      expect(result.current.predictions).toEqual([{ id: 99 }])
    );

    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/models/f2_to_f1'
    );
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/predictions/f2_to_f1/model1'
    );
  });

  it('handles refresh request failure', async () => {
    mockInitialLoad();

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    mockFetch.mockResolvedValueOnce({ ok: false, status: 500 });

    await act(() => result.current.refreshPredictions());

    await waitFor(() =>
      expect(result.current.error).toBe(
        'Could not refresh data. Please try again later.'
      )
    );
  });

  it('handles successful refresh with updated timestamps', async () => {
    mockInitialLoad();

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    const updatedStatus = {
      ...mockModelsResponse.system_status,
      last_scrape_predictions: '2023-01-02T00:00:00Z',
    };

    mockFetch
      .mockResolvedValueOnce({ ok: true, json: () => Promise.resolve({}) }) // POST refresh
      .mockResolvedValueOnce({
        ok: true,
        json: () =>
          Promise.resolve({
            ...mockModelsResponse,
            system_status: updatedStatus,
          }),
      }) // models poll
      .mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockPredictionsResponse),
      }); // predictions force-fetch

    await act(() => result.current.refreshPredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    expect(result.current.status?.last_scrape_predictions).toBe(
      '2023-01-02T00:00:00Z'
    );
    expect(mockFetch).toHaveBeenCalledWith(
      'http://localhost:8000/api/system/refresh/predictions',
      { method: 'POST' }
    );
  });

  it('handles refresh timeout after max attempts', async () => {
    mockInitialLoad();

    const { result } = renderHook(() => usePredictions());

    await waitFor(() => expect(result.current.loading).toBe(false));

    mockFetch.mockResolvedValueOnce({
      ok: true,
      json: () => Promise.resolve({}),
    });

    // 10 polls with unchanged timestamps
    for (let i = 0; i < 10; i++) {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockModelsResponse),
      });
    }

    await act(() => result.current.refreshPredictions());

    await waitFor(
      () => {
        expect(result.current.error).toBe('Update check timeout');
        expect(result.current.loading).toBe(false);
      },
      { timeout: 35000 }
    );
  }, 40000);

  it('uses VITE_API_URL env var', () => {
    import.meta.env.VITE_API_URL = 'https://api.example.com';
    mockInitialLoad();

    renderHook(() => usePredictions());

    expect(mockFetch).toHaveBeenCalledWith(
      'https://api.example.com/api/models/f3_to_f2'
    );

    import.meta.env.VITE_API_URL = undefined;
  });
});
