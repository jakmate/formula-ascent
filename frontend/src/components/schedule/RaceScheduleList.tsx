import { Calendar, MapPin } from 'lucide-react';
import { getFlagComponent } from '../../utils/flags';

interface SessionDetails {
  start?: string;
  time?: string;
}

interface Session {
  race?: SessionDetails;
}

export interface Race {
  slug?: string;
  round: number;
  name: string;
  location: string;
  sessions: Session;
}

interface RaceScheduleListProps {
  races: Race[];
  selectedSeries: string;
}

export const RaceScheduleList = ({
  races,
  selectedSeries,
}: RaceScheduleListProps) => {
  const userTimezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      timeZone: userTimezone,
      month: 'short',
      day: 'numeric',
    });
  };

  const formatTime = (dateString: string) => {
    return new Date(dateString).toLocaleTimeString('en-US', {
      timeZone: userTimezone,
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const isPastRace = (raceDate: string) => {
    if (!raceDate) return false;
    try {
      const now = new Date();
      const date = new Date(raceDate);
      if (raceDate.length === 10) {
        const today = new Date();
        today.setHours(0, 0, 0, 0);
        return date < today;
      }
      return date < now;
    } catch {
      return false;
    }
  };

  const getGrandPrixText = (series: string) => {
    return series === 'f1' ? 'GRAND PRIX' : 'Grand Prix';
  };

  const getRaceCardClasses = (past: boolean, isUpcoming: boolean) => {
    if (past) {
      return 'bg-[#0A0A0A] border-[#2A2A2A] opacity-60';
    } else if (isUpcoming) {
      return 'bg-[#151515] border-[#00FF66]';
    } else {
      return 'bg-[#151515] border-[#2A2A2A]';
    }
  };

  if (!races || races.length === 0) {
    return (
      <div className="text-center text-[#888888] font-mono py-8">
        <Calendar className="w-12 h-12 mx-auto mb-4 text-[#2A2A2A]" />
        <p>No races found for this series</p>
      </div>
    );
  }

  const firstUpcomingIndex = races.findIndex((race) => {
    const raceDate = race.sessions.race?.start;
    return raceDate ? !isPastRace(raceDate) : false;
  });

  return (
    <div className="space-y-2 mb-4">
      <div className="text-center text-[#888888] text-sm font-mono">
        Times shown in {userTimezone.replace('_', ' ')}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
        {races.map((race: Race, index: number) => {
          const raceSession = race.sessions.race;
          const raceDate = raceSession?.start;
          const isTBC = raceSession?.time === 'TBC';
          const past = raceDate ? isPastRace(raceDate) : false;
          const isUpcoming = index === firstUpcomingIndex;

          return (
            <div
              key={race.slug || index}
              className={`border p-4 transition-colors duration-100 ${getRaceCardClasses(past, isUpcoming)}`}
            >
              <div className="flex flex-col h-full">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[#888888] text-sm font-mono">
                    Round {race.round}
                  </span>
                  <div>
                    {past && (
                      <span className="text-xs bg-[#151515] text-[#FF0033] border border-[#FF0033] px-2 py-1 font-mono">
                        COMPLETED
                      </span>
                    )}
                    {isUpcoming && (
                      <span className="text-xs bg-[#151515] text-[#00FF66] border border-[#00FF66] px-2 py-1 font-mono">
                        NEXT RACE
                      </span>
                    )}
                  </div>
                </div>

                <h3 className="text-[#E0E0E0] font-mono font-semibold mb-2">
                  {race.name} {getGrandPrixText(selectedSeries)}
                </h3>

                <div className="flex items-center mb-3 text-[#888888] font-mono">
                  <MapPin className="w-4 h-4 mr-1" />
                  <span className="text-sm">{race.location}</span>
                  <div className="ml-2">{getFlagComponent(race.location)}</div>
                </div>

                {raceDate && (
                  <div className="mt-auto">
                    <div className="text-[#E0E0E0] font-mono">
                      {formatDate(raceDate)}
                    </div>
                    <div className="text-[#888888] text-sm font-mono">
                      {isTBC ? 'TBC' : formatTime(raceDate)}
                    </div>
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
