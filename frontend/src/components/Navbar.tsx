import { Calendar, Icon, Coffee, TrendingUp } from 'lucide-react';
import { motorRacingHelmet } from '@lucide/lab';
import { Link } from 'react-router-dom';

interface NavbarProps {
  activeView: 'predictions' | 'schedule';
}

export const Navbar = ({ activeView }: NavbarProps) => {
  const isActive = (view: string) => activeView === view;

  return (
    <nav className="bg-[#151515] border-b border-[#2A2A2A]">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          <div className="flex items-center space-x-2">
            <Icon
              iconNode={motorRacingHelmet}
              className="w-8 h-8 text-[#00FF66]"
            />
            <h1 className="text-2xl pl-4 font-bold text-[#E0E0E0] hidden sm:block font-mono">
              Formula Ascent
            </h1>
          </div>

          <div className="flex space-x-1">
            <Link
              to="/predictions"
              aria-label="View Promotions"
              className={`px-3 sm:px-6 py-2 font-medium transition-colors duration-100 flex items-center gap-2 font-mono ${
                isActive('predictions')
                  ? 'bg-[#2A2A2A] text-[#00FF66] border-l-2 border-[#00FF66]'
                  : 'text-[#E0E0E0] hover:bg-[#2A2A2A] hover:text-white'
              }`}
            >
              <TrendingUp className="w-4 h-4" aria-hidden="true" />
              <span className="hidden sm:inline">Promotions</span>
            </Link>

            <Link
              to="/schedule"
              aria-label="View Schedule"
              className={`px-3 sm:px-6 py-2 font-medium transition-colors duration-100 flex items-center gap-2 font-mono ${
                isActive('schedule')
                  ? 'bg-[#2A2A2A] text-[#00FF66] border-l-2 border-[#00FF66]'
                  : 'text-[#E0E0E0] hover:bg-[#2A2A2A] hover:text-white'
              }`}
            >
              <Calendar className="w-4 h-4" aria-hidden="true" />
              <span className="hidden sm:inline">Schedule</span>
            </Link>

            <a
              href="https://www.buymeacoffee.com/jakmate"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="Buy me a coffee - Support the project"
              className="px-3 sm:px-6 py-2 font-medium transition-colors duration-100 flex items-center gap-2 text-[#E0E0E0] hover:bg-[#2A2A2A] hover:text-white font-mono"
            >
              <Coffee className="w-4 h-4" aria-hidden="true" />
              <span className="hidden sm:inline">Coffee</span>
            </a>
          </div>
        </div>
      </div>
    </nav>
  );
};
