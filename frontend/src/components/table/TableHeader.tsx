import { ChevronUp, ChevronDown } from 'lucide-react';
import type { SortConfig, SortField } from '../../types/Sorting';

interface TableHeaderProps {
  field: SortField;
  sortConfig: SortConfig;
  onSort: (field: SortField) => void;
  children: React.ReactNode;
}

export const TableHeader = ({
  field,
  sortConfig,
  onSort,
  children,
}: TableHeaderProps) => {
  const isActive = sortConfig.field === field;

  const getSortIcon = () => {
    if (!isActive) return null;
    if (sortConfig.direction === 'asc') {
      return <ChevronUp className="w-4 h-4 ml-1 text-[#00FF66]" />;
    }
    return <ChevronDown className="w-4 h-4 ml-1 text-[#00FF66]" />;
  };

  const SortIcon = getSortIcon();

  return (
    <th
      className="p-4 font-mono font-semibold text-[#E0E0E0] cursor-pointer select-none border-b border-[#2A2A2A] hover:bg-[#2A2A2A]"
      onClick={() => onSort(field)}
    >
      <div className="flex items-center">
        {children}
        {SortIcon}
      </div>
    </th>
  );
};
