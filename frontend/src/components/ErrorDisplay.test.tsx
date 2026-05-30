import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import { ErrorDisplay } from './ErrorDisplay';

describe('ErrorDisplay', () => {
  it('renders the error message', () => {
    const errorMessage = 'Something went wrong';

    render(<ErrorDisplay error={errorMessage} />);

    expect(screen.getByText(errorMessage)).toBeInTheDocument();
  });

  it('renders different error messages', () => {
    const { rerender } = render(<ErrorDisplay error="Error 1" />);
    expect(screen.getByText('Error 1')).toBeInTheDocument();

    rerender(<ErrorDisplay error="Error 2" />);
    expect(screen.getByText('Error 2')).toBeInTheDocument();
  });
});
