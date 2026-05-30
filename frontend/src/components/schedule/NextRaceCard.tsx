import { Clock, MapPin, Calendar, Trophy, CheckCircle } from 'lucide-react';
import { useEffect, useState } from 'react';
import { getFlagComponent } from '../../utils/flags';

const sessionDisplayNames: Record<string, string> = {
  practice: 'PRACTICE',
  fp1: 'PRACTICE 1',
  fp2: 'PRACTICE 2',
  fp3: 'PRACTICE 3',
  qualifying: 'QUALIFYING',
  qualifying_group_a: 'QUALIFYING GROUP A',
  qualifying_group_b: 'QUALIFYING GROUP B',
  sprint_qualifying: 'SPRINT QUALIFYING',
  sprint: 'SPRINT',
  race: 'RACE',
};

interface SessionInfo {
  start: string;
  end?: string;
  time?: string;
}

interface NextSession {
  name: string;
  date: string;
  isTBC?: boolean;
}

interface NextRace {
  name: string;
  round: number;
  totalRounds?: number;
  location: string;
  sessions: Record<string, SessionInfo>;
  nextSession?: NextSession;
  seasonCompleted?: boolean;
}

interface NextRaceCardProps {
  nextRace: NextRace | null;
  userTimezone?: string;
}

export const NextRaceCard = ({ nextRace, userTimezone }: NextRaceCardProps) => {
  const [currentTime, setCurrentTime] = useState(new Date());
  const [countdown, setCountdown] = useState<string | null>(null);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentTime(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (nextRace?.nextSession && !nextRace.seasonCompleted) {
      const updateCountdown = () => {
        let sessionTime: Date;
        const sessionDateStr = nextRace.nextSession!.date;

        // Handle both date-only strings (TBC) and full datetime strings
        if (sessionDateStr.length === 10) {
          // Date-only format (YYYY-MM-DD) - create date at start of day
          const [year, month, day] = sessionDateStr.split('-');
          sessionTime = new Date(
            Number.parseInt(year),
            Number.parseInt(month) - 1,
            Number.parseInt(day)
          );
        } else {
          // Full datetime string
          sessionTime = new Date(sessionDateStr);
        }
        const diff = sessionTime.getTime() - currentTime.getTime();
        if (diff <= 0) {
          setCountdown('LIVE NOW');
          return;
        }
        const days = Math.floor(diff / (1000 * 60 * 60 * 24));
        const hours = Math.floor(
          (diff % (1000 * 60 * 60 * 24)) / (1000 * 60 * 60)
        );
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);
        if (days > 0) setCountdown(`${days}d ${hours}h`);
        else if (hours > 0) setCountdown(`${hours}h ${minutes}m`);
        else setCountdown(`${minutes}m ${seconds}s`);
      };
      updateCountdown();
    } else if (nextRace?.seasonCompleted) {
      const apply = () => setCountdown('SEASON COMPLETED');
      apply();
    }
  }, [currentTime, nextRace]);

  if (!nextRace) return null;

  const isSeasonCompleted = nextRace.seasonCompleted || false;

  // Format date - times are already converted by backend
  const formatDate = (dateString: string) => {
    let date: Date;
    if (dateString.length === 10) {
      const [year, month, day] = dateString.split('-');
      date = new Date(
        Number.parseInt(year),
        Number.parseInt(month) - 1,
        Number.parseInt(day)
      );
    } else {
      date = new Date(dateString);
    }
    if (Number.isNaN(date.getTime())) return 'Date TBC';
    const weekday = date.toLocaleDateString('en-GB', { weekday: 'long' });
    const day = date.getDate();
    const month = date.toLocaleDateString('en-GB', { month: 'long' });
    const year = date.getFullYear();
    return `${weekday}, ${day} ${month}, ${year}`;
  };

  const formatTime = (dateString: string) => {
    try {
      return new Date(dateString).toLocaleTimeString(undefined, {
        hour: '2-digit',
        minute: '2-digit',
      });
    } catch {
      return 'TBC';
    }
  };

  // Format short date
  const formatShortDate = (dateString: string) => {
    let date: Date;
    if (dateString.length === 10) {
      const [year, month, day] = dateString.split('-');
      date = new Date(
        Number.parseInt(year),
        Number.parseInt(month) - 1,
        Number.parseInt(day)
      );
    } else {
      date = new Date(dateString);
    }
    if (Number.isNaN(date.getTime())) return 'TBC';
    const weekday = date.toLocaleDateString('en-GB', { weekday: 'short' });
    const day = date.getDate();
    const month = date.toLocaleDateString('en-GB', { month: 'short' });
    return `${weekday}, ${day} ${month}`;
  };

  // Determine session status
  const getSessionStatus = (sessionInfo: SessionInfo) => {
    if (sessionInfo.time === 'TBC') return 'tbc';
    try {
      const start = new Date(sessionInfo.start);
      const end = sessionInfo.end ? new Date(sessionInfo.end) : null;
      if (currentTime < start) return 'upcoming';
      if (end && currentTime <= end) return 'live';
      return 'completed';
    } catch {
      return 'tbc';
    }
  };

  const displayTimezone = userTimezone?.replaceAll('_', ' ') || 'Local Time';

  return (
    <div
      className={`bg-[#151515] border mb-8 ${isSeasonCompleted ? 'border-[#00FF66]' : 'border-[#2A2A2A]'}`}
    >
      {/* Top accent line flat */}
      <div
        className={`h-1 w-full ${isSeasonCompleted ? 'bg-[#00FF66]' : 'bg-[#2A2A2A]'}`}
      ></div>

      <div className="p-6">
        <div className="flex flex-col items-center text-center mb-8">
          <div className="flex items-center mb-3">
            {isSeasonCompleted ? (
              <CheckCircle className="w-8 h-8 text-[#00FF66] mr-2" />
            ) : (
              <Trophy className="w-6 h-6 md:w-8 md:h-8 text-[#FFCC00] mr-2" />
            )}
            <h2 className="text-lg sm:text-xl md:text-2xl font-mono font-bold text-[#E0E0E0]">
              {isSeasonCompleted ? 'LAST RACE' : 'NEXT RACE'}: {nextRace.name}{' '}
              GP
            </h2>
            <div className="ml-2 mt-1">
              {getFlagComponent(nextRace.location)}
            </div>
          </div>
          <div
            className={`px-4 py-1.5 border font-mono text-xs sm:text-sm ${isSeasonCompleted ? 'border-[#00FF66] text-[#00FF66]' : 'border-[#2A2A2A] text-[#888888]'}`}
          >
            Round {nextRace.round} of {nextRace.totalRounds || '?'}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8 justify-items-center">
          <div className="flex flex-col items-center max-w-xs text-center">
            <div className="flex items-center mb-2">
              <MapPin className="w-6 h-6 mr-2 text-[#FF0033]" />
              <h3 className="text-[#888888] text-xs sm:text-sm font-mono">
                Location
              </h3>
            </div>
            <p className="text-[#E0E0E0] sm:text-xl font-mono font-semibold">
              {nextRace.location}
            </p>
          </div>
          <div className="flex flex-col items-center max-w-xs text-center">
            <div className="flex items-center mb-2">
              <Calendar className="w-6 h-6 mr-2 text-[#00FF66]" />
              <h3 className="text-[#888888] text-xs sm:text-sm font-mono">
                Race Day
              </h3>
            </div>
            <p className="text-[#E0E0E0] sm:text-xl font-mono font-semibold">
              {formatDate(nextRace.sessions.race.start)}
            </p>
          </div>
          {(nextRace.nextSession || isSeasonCompleted) && (
            <div className="flex flex-col items-center max-w-xs text-center">
              <div className="flex items-center mb-2">
                <Clock className="w-6 h-6 mr-2 text-[#FFCC00]" />
                <h3 className="text-[#888888] text-xs sm:text-sm font-mono">
                  {isSeasonCompleted ? 'Status' : 'Next Session'}
                </h3>
              </div>
              <p className="text-[#E0E0E0] sm:text-xl font-mono font-semibold">
                {isSeasonCompleted ? (
                  <span className="text-[#00FF66] font-bold">
                    SEASON COMPLETED
                  </span>
                ) : (
                  <>
                    {nextRace.nextSession!.name.toUpperCase()}
                    {nextRace.nextSession!.isTBC} -
                    {countdown ? (
                      <span
                        className={`inline-block ml-1 font-bold ${countdown === 'LIVE NOW' ? 'text-[#FF0033]' : 'text-[#00FF66]'}`}
                      >
                        {countdown}
                      </span>
                    ) : (
                      <span className="inline-block text-[#888888] ml-1 font-bold">
                        Loading...
                      </span>
                    )}
                  </>
                )}
              </p>
            </div>
          )}
        </div>

        {nextRace.sessions && (
          <div className="bg-[#0A0A0A] border border-[#2A2A2A] p-6">
            <div className="flex flex-col items-center mb-5">
              <h3 className="text-[#888888] text-sm font-mono mb-1 uppercase tracking-wider">
                SESSION TIMETABLE
              </h3>
              <div className="text-[#2A2A2A] text-xs font-mono">
                All times in {displayTimezone}
              </div>
            </div>
            <div className="flex flex-wrap justify-center gap-2 sm:gap-4">
              {Object.entries(nextRace.sessions).map(
                ([sessionKey, sessionInfo]) => {
                  const sessionName =
                    sessionDisplayNames[sessionKey] || sessionKey.toUpperCase();
                  const isTBC = sessionInfo.time === 'TBC';
                  const status = isSeasonCompleted
                    ? 'completed'
                    : getSessionStatus(sessionInfo);

                  let borderColor = 'border-[#2A2A2A]';
                  let textAccent = 'text-[#888888]';
                  let statusLabel = null;

                  if (status === 'live') {
                    borderColor = 'border-l-[3px] border-l-[#FF0033]';
                    textAccent = 'text-[#FF0033]';
                    statusLabel = (
                      <div className="mt-1 text-xs font-bold text-[#FF0033]">
                        LIVE NOW
                      </div>
                    );
                  } else if (status === 'upcoming') {
                    borderColor = 'border-l-[3px] border-l-[#00FF66]';
                    textAccent = 'text-[#00FF66]';
                    statusLabel = (
                      <div className="mt-1 text-xs font-bold text-[#00FF66]">
                        UPCOMING
                      </div>
                    );
                  } else if (status === 'completed') {
                    borderColor = 'border-l-[3px] border-l-[#2A2A2A]';
                    textAccent = 'text-[#888888]';
                    statusLabel = (
                      <div className="mt-1 text-xs font-bold text-[#2A2A2A]">
                        COMPLETED
                      </div>
                    );
                  } else if (status === 'tbc') {
                    borderColor = 'border-l-[3px] border-l-[#FFCC00]';
                    textAccent = 'text-[#FFCC00]';
                    statusLabel = (
                      <div className="mt-1 text-xs font-bold text-[#FFCC00]">
                        TBC
                      </div>
                    );
                  }

                  return (
                    <div
                      key={sessionKey}
                      className={`bg-[#151515] border border-[#2A2A2A] ${borderColor} p-3 sm:p-4 flex flex-col items-center min-w-[80px] sm:min-w-[100px] md:min-w-[120px] lg:min-w-[160px]`}
                    >
                      <div
                        className={`text-sm lg:text-lg font-mono font-bold uppercase mb-1 ${textAccent}`}
                      >
                        {sessionName}
                      </div>
                      {isTBC ? (
                        <>
                          <div className="text-[#E0E0E0] text-lg sm:text-xl font-mono font-bold mb-1">
                            TBC
                          </div>
                          <div className="mt-1 text-center">
                            <div className="text-[#888888] text-xs sm:text-sm font-mono">
                              {formatShortDate(sessionInfo.start)}
                            </div>
                            <div className="mt-1 text-xs font-bold text-[#FFCC00]">
                              TBC
                            </div>
                          </div>
                        </>
                      ) : (
                        <>
                          <div className="flex flex-col items-center">
                            <div className="text-[#E0E0E0] text-sm sm:text-lg font-mono font-bold">
                              {formatTime(sessionInfo.start)}
                            </div>
                            {sessionInfo.end && (
                              <>
                                <div className="text-[#888888] text-xs md:text-sm font-mono">
                                  to
                                </div>
                                <div className="text-[#E0E0E0] text-sm md:text-lg font-mono font-bold">
                                  {formatTime(sessionInfo.end)}
                                </div>
                              </>
                            )}
                          </div>
                          <div className="mt-1 sm:mt-2 text-center">
                            <div className="text-[#888888] text-xs sm:text-sm font-mono">
                              {formatShortDate(sessionInfo.start)}
                            </div>
                            {statusLabel}
                          </div>
                        </>
                      )}
                    </div>
                  );
                }
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
