import React, { useState, useRef, useEffect } from 'react';
import type { Driver } from '../../types/Driver';
import { getFlagComponent } from '../../utils/flags';

interface DriverHoverCardProps {
  driver: Driver;
  children: React.ReactNode;
}

export const DriverHoverCard = ({ driver, children }: DriverHoverCardProps) => {
  const [isVisible, setIsVisible] = useState(false);
  const [position, setPosition] = useState<'top' | 'bottom'>('bottom');
  const [cardHeight, setCardHeight] = useState(500);
  const timeoutIdRef = useRef<number | null>(null);
  const cardRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLDivElement>(null);

  // Update card height when it becomes visible
  useEffect(() => {
    if (isVisible && cardRef.current) {
      setCardHeight(cardRef.current.offsetHeight);
    }
  }, [isVisible]);

  const handleMouseEnter = () => {
    // Clear any pending hide operations
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }

    // Calculate position before showing
    if (triggerRef.current) {
      const rect = triggerRef.current.getBoundingClientRect();
      const spaceBelow = window.innerHeight - rect.bottom;
      const spaceAbove = rect.top;
      // Use estimated height for initial calculation
      setPosition(
        spaceBelow < cardHeight && spaceAbove > spaceBelow ? 'top' : 'bottom'
      );
    }

    // Show after delay
    timeoutIdRef.current = setTimeout(() => setIsVisible(true), 300);
  };

  const handleMouseLeave = () => {
    if (timeoutIdRef.current) {
      clearTimeout(timeoutIdRef.current);
      timeoutIdRef.current = null;
    }
    setIsVisible(false);
  };

  const driverInfo = {
    fullName: driver.driver,
    team: driver.team,
    nationality: driver.nationality,
    dateOfBirth: driver.dob,
    ...(typeof driver.age === 'number' ? { age: Math.floor(driver.age) } : {}),
    seasonWins: driver.wins,
    seasonPodiums: driver.podiums,
    photo: `https://ui-avatars.com/api/?name=${driver.driver
      .split(' ')
      .map((n) => n[0])
      .join('')}&background=0A0A0A&color=00FF66`,
  };

  const getExperienceDisplay = () => {
    if (driver.experience === 0) return 'Rookie';
    if (driver.experience === 1) return '1 year';
    return `${driver.experience} years`;
  };

  return (
    <span
      className="relative inline-block"
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      onFocus={handleMouseEnter}
      onBlur={handleMouseLeave}
      ref={triggerRef}
      role="tooltip"
      aria-label={`View details for ${driver.driver}`}
    >
      <div ref={triggerRef}>{children}</div>

      {isVisible && (
        <div
          ref={cardRef}
          className={`absolute z-50 left-0 w-80 bg-[#151515] border border-[#2A2A2A] p-4 ${
            position === 'top' ? 'bottom-full mb-2' : 'top-full mt-2'
          }`}
        >
          <div className="flex items-start gap-4">
            <div className="relative">
              <img
                src={driverInfo.photo}
                alt={driverInfo.fullName}
                className="w-16 h-16 object-cover bg-[#0A0A0A] border border-[#2A2A2A]"
              />
            </div>

            <div className="flex-1">
              <div className="flex items-center gap-2 mb-2">
                <h3 className="text-[#E0E0E0] font-mono font-semibold text-lg">
                  {driverInfo.fullName}
                </h3>
                {getFlagComponent(driverInfo.nationality)}
              </div>

              <div className="space-y-1 text-sm font-mono">
                <div className="flex justify-between">
                  <span className="text-[#888888]">Team:</span>
                  <span className="text-[#E0E0E0]">{driverInfo.team}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#888888]">DoB:</span>
                  <span className="text-[#E0E0E0]">
                    {driverInfo.dateOfBirth}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#888888]">Age:</span>
                  <span className="text-[#E0E0E0]">{driverInfo.age}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-[#888888]">Experience:</span>
                  <span className="text-[#E0E0E0]">
                    {getExperienceDisplay()}
                  </span>
                </div>
              </div>

              <div className="mt-3 pt-2 border-t border-[#2A2A2A]">
                <div className="text-xs text-[#888888] font-mono">
                  Current Season
                </div>
                <div className="flex justify-between text-sm font-mono mt-1">
                  <span className="text-[#888888]">Position:</span>
                  <span className="text-[#E0E0E0]">
                    {driver.position === -1 ? '-' : `#${driver.position}`}
                  </span>
                </div>
                <div className="flex justify-between text-sm font-mono">
                  <span className="text-[#888888]">Points:</span>
                  <span className="text-[#E0E0E0]">
                    {driver.points.toFixed(1)}
                  </span>
                </div>
                <div className="flex justify-between text-sm font-mono">
                  <span className="text-[#888888]">Wins:</span>
                  <span className="text-[#00FF66]">
                    {driverInfo.seasonWins}
                  </span>
                </div>
                <div className="flex justify-between text-sm font-mono">
                  <span className="text-[#888888]">Podiums:</span>
                  <span className="text-[#FFCC00]">
                    {driverInfo.seasonPodiums}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </span>
  );
};
