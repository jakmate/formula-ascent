import { RefreshCw } from 'lucide-react';
import { NextRaceCard } from './NextRaceCard';
import { RaceScheduleList } from './RaceScheduleList';
import { useSchedule } from '../../hooks/useSchedule';
import { Header } from '../Header';
import { ErrorDisplay } from '../ErrorDisplay';

export const Schedule = () => {
  const {
    races,
    nextRace,
    selectedSeries,
    setSelectedSeries,
    series,
    loading,
    error,
    refreshSchedule,
  } = useSchedule();

  return (
    <div className="w-full">
      <Header
        title="Race Schedule"
        rightContent={
          <div className="flex flex-col sm:flex-row gap-3">
            <label className="flex flex-col gap-2">
              <span className="text-sm text-[#888888] sr-only font-mono">
                Series
              </span>
              <select
                value={selectedSeries}
                onChange={(e) => setSelectedSeries(e.target.value)}
                disabled={loading}
                className="px-4 py-2 bg-[#151515] border border-[#2A2A2A] text-[#E0E0E0] font-mono focus:outline-none focus:border-[#00FF66]"
              >
                {series.map((s) => (
                  <option
                    key={s.value}
                    value={s.value}
                    className="bg-[#0A0A0A] text-[#E0E0E0]"
                  >
                    {s.label}
                  </option>
                ))}
              </select>
            </label>

            <button
              onClick={refreshSchedule}
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
      />

      {error && <ErrorDisplay error={error} />}

      {nextRace && <NextRaceCard nextRace={nextRace} />}

      {/* Flat panel: no blur, no radius, no gradient, no cyan border */}
      <div className="bg-[#151515] border border-[#2A2A2A]">
        <h2 className="text-lg md:text-xl font-mono font-semibold text-[#E0E0E0] p-6 border-b border-[#2A2A2A]">
          Full Season Schedule
        </h2>

        {loading ? (
          <div className="p-12 text-center text-[#E0E0E0] font-mono">
            <RefreshCw className="w-8 h-8 animate-spin mx-auto mb-4 text-[#00FF66]" />
            <p>Loading schedule...</p>
          </div>
        ) : (
          <div className="p-4">
            <RaceScheduleList races={races} selectedSeries={selectedSeries} />
          </div>
        )}
      </div>
    </div>
  );
};
