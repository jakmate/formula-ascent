import {
  Calendar,
  RefreshCw,
  Target,
  TrendingUp,
  UserRound,
} from 'lucide-react';
import { ErrorDisplay } from '../ErrorDisplay';
import { TableContent } from './TableContent';
import { usePredictions, type SeriesType } from '../../hooks/usePredictions';
import { Header } from '../Header';

interface SeriesOption {
  value: string;
  label: string;
}

interface BasePredictionsTableProps {
  defaultSeries: SeriesType;
  seriesOptions: SeriesOption[];
  getTitle: (selectedSeries: SeriesType) => string;
  getDescription: (selectedSeries: SeriesType) => string;
}

export const BasePredictionsTable = ({
  defaultSeries,
  seriesOptions,
  getTitle,
  getDescription,
}: BasePredictionsTableProps) => {
  const {
    selectedModel,
    setSelectedModel,
    models,
    loading,
    status,
    error,
    refreshPredictions,
    currentPredictions,
    series,
    setSeries,
  } = usePredictions(defaultSeries);

  const getPredictionsDisplay = () => {
    if (loading && currentPredictions.length === 0) {
      return (
        <div className="p-12 text-center text-[#E0E0E0] font-mono">
          <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-[#00FF66]" />
          <p>Loading predictions...</p>
        </div>
      );
    } else if (currentPredictions.length === 0) {
      return (
        <div className="p-12 text-center text-[#E0E0E0] font-mono">
          <p>No predictions available. Select a model and refresh data.</p>
        </div>
      );
    } else {
      return <TableContent predictions={currentPredictions} />;
    }
  };

  return (
    <div className="w-full">
      <Header
        title={getTitle(series)}
        description={getDescription(series)}
        rightContent={
          <div className="flex flex-col md:flex-row gap-3">
            <label className="flex flex-col gap-2">
              <span className="text-sm text-[#888888] sr-only font-mono">
                Series
              </span>
              <select
                value={series}
                onChange={(e) => setSeries(e.target.value as SeriesType)}
                disabled={loading}
                className="px-4 py-2 bg-[#151515] border border-[#2A2A2A] text-[#E0E0E0] font-mono focus:outline-none focus:border-[#00FF66]"
              >
                {seriesOptions.map((option) => (
                  <option
                    key={option.value}
                    value={option.value}
                    className="bg-[#0A0A0A] text-[#E0E0E0]"
                  >
                    {option.label}
                  </option>
                ))}
              </select>
            </label>

            <label className="flex flex-col gap-2">
              <span className="text-sm text-[#888888] sr-only font-mono">
                Model
              </span>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                disabled={loading}
                className="px-4 py-2 bg-[#151515] border border-[#2A2A2A] text-[#E0E0E0] font-mono focus:outline-none focus:border-[#00FF66]"
              >
                <option value="" className="bg-[#0A0A0A]">
                  Select Model
                </option>
                {models.map((model) => (
                  <option
                    key={model}
                    value={model}
                    className="bg-[#0A0A0A] text-[#E0E0E0]"
                  >
                    {model}
                  </option>
                ))}
              </select>
            </label>

            <button
              onClick={() => {
                void refreshPredictions();
              }}
              disabled={loading}
              className="px-6 py-2 bg-[#2A2A2A] hover:bg-[#00FF66] hover:text-[#0A0A0A] disabled:opacity-50 text-[#00FF66] font-mono transition-colors duration-100 flex items-center gap-2 border border-[#2A2A2A]"
            >
              <RefreshCw
                className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`}
              />
              {loading ? 'Updating...' : 'Refresh'}
            </button>
          </div>
        }
        bottomContent={
          status && (
            <div className="flex flex-wrap gap-4 text-[#00FF66] font-mono text-sm">
              {status.last_scrape_predictions && (
                <div className="flex items-center gap-1">
                  <Calendar className="w-4 h-4" />
                  Last scrape:{' '}
                  {new Date(status.last_scrape_predictions).toLocaleString()}
                </div>
              )}
              {status.last_training && (
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-4 h-4" />
                  Last training:{' '}
                  {new Date(status.last_training).toLocaleString()}
                </div>
              )}
              <div className="flex items-center gap-1">
                <Target className="w-4 h-4" />
                Models: {status.models_available?.[series]?.length || 0}
              </div>
              <div className="flex items-center gap-1">
                <UserRound className="w-4 h-4" />
                Drivers: {currentPredictions.length}
              </div>
            </div>
          )
        }
      />

      {error && <ErrorDisplay error={error} />}

      <div className="bg-[#151515] border border-[#2A2A2A] overflow-hidden">
        {getPredictionsDisplay()}
      </div>
    </div>
  );
};
