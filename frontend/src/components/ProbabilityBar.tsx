interface ProbabilityBarProps {
  percentage: number;
}

export const ProbabilityBar = ({ percentage }: ProbabilityBarProps) => {
  const getColor = () => {
    if (percentage > 70) return '#00FF66';
    if (percentage > 40) return '#FFCC00';
    return '#FF0033';
  };

  return (
    <div className="flex items-center gap-2">
      {/* Container: flat dark, no radius */}
      <div className="w-20 h-2 bg-[#2A2A2A] overflow-hidden">
        <div
          className="h-full transition-all duration-500"
          style={{
            width: `${Math.min(percentage, 100)}%`,
            backgroundColor: getColor(),
          }}
        />
      </div>
      <span className="text-[#E0E0E0] text-sm font-mono">
        {percentage.toFixed(1)}%
      </span>
    </div>
  );
};
