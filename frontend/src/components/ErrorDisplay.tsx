interface ErrorDisplayProps {
  error: string;
}

export const ErrorDisplay = ({ error }: ErrorDisplayProps) => (
  <div className="bg-[#151515] border-l-4 border-[#FF0033] p-4 mb-6 text-[#E0E0E0] font-mono">
    {error}
  </div>
);
