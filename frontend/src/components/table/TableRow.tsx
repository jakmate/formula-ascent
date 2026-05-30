import type { Driver } from '../../types/Driver';
import { ProbabilityBar } from '../ProbabilityBar';
import { DriverHoverCard } from './DriverHoverCard';

interface TableRowProps {
  driver: Driver;
  className?: string;
}

export const TableRow = ({ driver, className = '' }: TableRowProps) => {
  const empiricalPercentage = driver.empirical_percentage ?? 0;
  const formatPercentage = (value: number) => (value * 100).toFixed(1) + '%';

  const baseClasses = `border-t border-[#2A2A2A] hover:bg-[#2A2A2A] transition-colors`;

  return (
    <tr className={`${baseClasses} ${className}`}>
      <td className="p-4">
        <DriverHoverCard driver={driver}>
          <div className="flex items-center gap-3">
            <span className="text-[#E0E0E0] font-mono font-medium hover:text-[#00FF66] transition-colors cursor-pointer">
              {driver.driver}
            </span>
          </div>
        </DriverHoverCard>
      </td>
      <td className="p-4">
        <span className="text-[#E0E0E0] font-mono">
          {driver.position === -1 ? '-' : `#${driver.position}`}
        </span>
      </td>
      <td className="p-4">
        <span className="text-[#E0E0E0] font-mono">
          {driver.points.toFixed(1)}
        </span>
      </td>
      <td className="p-4">
        <span className="text-[#E0E0E0] font-mono">
          {formatPercentage(driver.win_rate)}
        </span>
      </td>
      <td className="p-4">
        <span className="text-[#E0E0E0] font-mono">
          {formatPercentage(driver.dnf_rate)}
        </span>
      </td>
      <td className="p-4">
        <span className="text-[#E0E0E0] font-mono">
          {formatPercentage(driver.participation_rate)}
        </span>
      </td>
      <td className="p-4">
        <span className="text-[#E0E0E0] font-mono">
          {driver.experience} years
        </span>
      </td>
      <td className="p-4">
        <ProbabilityBar percentage={empiricalPercentage} />
      </td>
    </tr>
  );
};
