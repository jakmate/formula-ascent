import type { ReactNode } from 'react';

interface HeaderProps {
  title: string;
  description?: string;
  leftContent?: ReactNode;
  rightContent?: ReactNode;
  bottomContent?: ReactNode;
}

export const Header = ({
  title,
  description,
  leftContent,
  rightContent,
  bottomContent,
}: HeaderProps) => (
  <div className="bg-[#151515] border border-[#2A2A2A] p-6 mb-6">
    <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
      <div>
        <h1 className="text-xl md:text-2xl font-mono font-bold text-[#E0E0E0] mb-2">
          {title}
        </h1>
        {description && (
          <p className="text-sm md:text-base text-[#888888] font-mono">
            {description}
          </p>
        )}
        {leftContent}
      </div>

      {rightContent && (
        <div className="text-xs md:text-base flex flex-col sm:flex-row gap-3">
          {rightContent}
        </div>
      )}
    </div>

    {bottomContent && (
      <div className="text-xs md:text-base mt-4 border-t border-[#2A2A2A] pt-4">
        {bottomContent}
      </div>
    )}
  </div>
);
