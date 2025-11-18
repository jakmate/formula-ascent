import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ErrorDisplay } from './ErrorDisplay';

describe('ErrorDisplay', () => {
  it('renders the error message', () => {
    const errorMessage = 'Something went wrong';

    render(<ErrorDisplay error={errorMessage} />);

    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('applies correct CSS classes for styling', () => {
    const errorMessage = 'Test error';

    const { container } = render(<ErrorDisplay error={errorMessage} />);
    const errorDiv = container.firstChild as HTMLElement;

    expect(errorDiv).toHaveClass(
      'bg-red-500/20',
      'border',
      'border-red-500/50',
      'rounded-lg',
      'p-4',
      'mb-6',
      'text-red-200',
      'backdrop-blur-sm'
    );
  });

  it('renders different error messages', () => {
    const { rerender } = render(<ErrorDisplay error="Error 1" />);
    expect(screen.getByText('Error 1')).toBeInTheDocument();

    rerender(<ErrorDisplay error="Error 2" />);
    expect(screen.getByText('Error 2')).toBeInTheDocument();
  });
});
