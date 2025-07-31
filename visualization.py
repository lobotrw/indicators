"""
Visualization module for backtesting results with professional scoring zones
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime

class BacktestVisualizer:
    """Create professional visualizations for backtesting results"""
    
    def __init__(self):
        # Set style
        plt.style.use('seaborn-v0_8')
        self.zone_colors = {
            'green': '#2E8B57',    # Sea Green
            'yellow': '#FFD700',   # Gold
            'red': '#DC143C'       # Crimson
        }
    
    def plot_optimization_heatmap(self, results_df, metric='composite_score'):
        """Create heatmap showing parameter optimization results"""
        
        # Pivot data for heatmap
        pivot_data = results_df.pivot_table(
            values=metric,
            index='sma_period',
            columns='ema_period',
            aggfunc='mean'
        )
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(
            pivot_data,
            annot=True,
            fmt='.2f',
            cmap='RdYlGn',
            center=pivot_data.mean().mean(),
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        plt.title(f'Parameter Optimization Heatmap - {metric.replace("_", " ").title()}')
        plt.xlabel('EMA Period')
        plt.ylabel('SMA Period')
        plt.tight_layout()
        
        filename = f'optimization_heatmap_{metric}_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_performance_zones(self, results_df):
        """Create scatter plot showing performance zones for key metrics"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Performance Zones Analysis', fontsize=16, fontweight='bold')
        
        # Profit Factor vs Sharpe Ratio
        ax1 = axes[0, 0]
        for zone in ['green', 'yellow', 'red']:
            zone_data = results_df[results_df['profit_factor_zone'] == zone]
            ax1.scatter(
                zone_data['profit_factor'],
                zone_data['sharpe_ratio'],
                c=self.zone_colors[zone],
                label=f'{zone.title()} Zone',
                alpha=0.7,
                s=50
            )
        
        ax1.axhline(y=2, color='gray', linestyle='--', alpha=0.5, label='Sharpe > 2')
        ax1.axvline(x=4, color='gray', linestyle='--', alpha=0.5, label='PF > 4')
        ax1.set_xlabel('Profit Factor')
        ax1.set_ylabel('Sharpe Ratio')
        ax1.set_title('Profit Factor vs Sharpe Ratio')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Win Rate vs Total Trades
        ax2 = axes[0, 1]
        for zone in ['green', 'yellow', 'red']:
            zone_data = results_df[results_df['win_rate_zone'] == zone]
            ax2.scatter(
                zone_data['total_trades'],
                zone_data['win_rate'],
                c=self.zone_colors[zone],
                label=f'{zone.title()} Zone',
                alpha=0.7,
                s=50
            )
        
        ax2.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='Win Rate > 50%')
        ax2.axvline(x=45, color='gray', linestyle='--', alpha=0.5, label='Trades 45-105')
        ax2.axvline(x=105, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Total Trades')
        ax2.set_ylabel('Win Rate (%)')
        ax2.set_title('Win Rate vs Total Trades')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Sortino vs Drawdown
        ax3 = axes[1, 0]
        for zone in ['green', 'yellow', 'red']:
            zone_data = results_df[results_df['sortino_ratio_zone'] == zone]
            ax3.scatter(
                zone_data['intra_trade_max_dd'],
                zone_data['sortino_ratio'],
                c=self.zone_colors[zone],
                label=f'{zone.title()} Zone',
                alpha=0.7,
                s=50
            )
        
        ax3.axhline(y=2.9, color='gray', linestyle='--', alpha=0.5, label='Sortino > 2.9')
        ax3.axvline(x=25, color='gray', linestyle='--', alpha=0.5, label='DD < 25%')
        ax3.set_xlabel('Intra-Trade Max Drawdown (%)')
        ax3.set_ylabel('Sortino Ratio')
        ax3.set_title('Sortino Ratio vs Max Drawdown')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Composite Score Distribution
        ax4 = axes[1, 1]
        ax4.hist(results_df['composite_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax4.axvline(x=results_df['composite_score'].mean(), color='red', linestyle='--', 
                   label=f'Mean: {results_df["composite_score"].mean():.2f}')
        ax4.set_xlabel('Composite Score')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Composite Score Distribution')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'performance_zones_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def plot_equity_curve(self, equity_curve, title="Strategy Equity Curve"):
        """Plot equity curve with drawdown"""
        
        if len(equity_curve) == 0:
            print("No equity curve data available")
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), height_ratios=[2, 1])
        
        # Equity curve
        ax1.plot(equity_curve['Date'], equity_curve['Equity'], 
                linewidth=2, color='blue', label='Portfolio Value')
        ax1.fill_between(equity_curve['Date'], equity_curve['Equity'], 
                        alpha=0.3, color='blue')
        
        ax1.set_title(title, fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format y-axis
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Drawdown
        equity_values = equity_curve['Equity'].values
        rolling_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - rolling_max) / rolling_max * 100
        
        ax2.fill_between(equity_curve['Date'], drawdowns, 0, 
                        color='red', alpha=0.3, label='Drawdown')
        ax2.plot(equity_curve['Date'], drawdowns, color='red', linewidth=1)
        
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        filename = f'equity_curve_{datetime.now().strftime("%Y%m%d_%H%M")}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        return filename
    
    def create_performance_report(self, results_df, top_n=10):
        """Create a comprehensive performance report"""
        
        print("=" * 80)
        print("BITCOIN BACKTESTING PERFORMANCE REPORT")
        print("=" * 80)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Total Strategies Tested: {len(results_df)}")
        print()
        
        # Zone distribution
        print("PERFORMANCE ZONE DISTRIBUTION:")
        print("-" * 40)
        
        for metric in ['profit_factor', 'sharpe_ratio', 'sortino_ratio', 'win_rate']:
            zone_col = f'{metric}_zone'
            if zone_col in results_df.columns:
                zone_counts = results_df[zone_col].value_counts()
                print(f"{metric.replace('_', ' ').title()}:")
                for zone in ['green', 'yellow', 'red']:
                    count = zone_counts.get(zone, 0)
                    pct = (count / len(results_df)) * 100
                    print(f"  {zone.title()}: {count} ({pct:.1f}%)")
                print()
        
        # Top performers
        print(f"TOP {top_n} STRATEGIES:")
        print("-" * 40)
        
        display_cols = [
            'ema_period', 'sma_period', 'rsi_period', 'composite_score',
            'total_return', 'profit_factor', 'sharpe_ratio', 'win_rate', 'total_trades'
        ]
        
        top_strategies = results_df.head(top_n)[display_cols].copy()
        
        # Format for display
        top_strategies['total_return'] = top_strategies['total_return'].map('{:.1f}%'.format)
        top_strategies['profit_factor'] = top_strategies['profit_factor'].map('{:.2f}'.format)
        top_strategies['sharpe_ratio'] = top_strategies['sharpe_ratio'].map('{:.2f}'.format)
        top_strategies['win_rate'] = top_strategies['win_rate'].map('{:.1f}%'.format)
        top_strategies['composite_score'] = top_strategies['composite_score'].map('{:.2f}'.format)
        
        print(top_strategies.to_string(index=False))
        print()
        
        # Best strategy detailed analysis
        best = results_df.iloc[0]
        print("BEST STRATEGY DETAILED ANALYSIS:")
        print("-" * 40)
        print(f"Parameters: EMA={best['ema_period']}, SMA={best['sma_period']}, RSI={best['rsi_period']}")
        print(f"Composite Score: {best['composite_score']:.2f}/3.00")
        print()
        print("Performance Metrics:")
        
        metrics_display = [
            ('Total Return', f"{best['total_return']:.1f}%"),
            ('Profit Factor', f"{best['profit_factor']:.2f} ({best['profit_factor_zone'].upper()})"),
            ('Sharpe Ratio', f"{best['sharpe_ratio']:.2f} ({best['sharpe_ratio_zone'].upper()})"),
            ('Sortino Ratio', f"{best['sortino_ratio']:.2f} ({best['sortino_ratio_zone'].upper()})"),
            ('Win Rate', f"{best['win_rate']:.1f}% ({best['win_rate_zone'].upper()})"),
            ('Total Trades', f"{best['total_trades']} ({best['total_trades_zone'].upper()})"),
            ('Max Drawdown', f"{best['max_drawdown']:.1f}%"),
            ('Omega Ratio', f"{best['omega_ratio']:.2f} ({best['omega_ratio_zone'].upper()})")
        ]
        
        for metric, value in metrics_display:
            print(f"  {metric}: {value}")
        
        print()
        print("=" * 80)

def main():
    """Example usage of the visualizer"""
    
    # This would typically be called from the main backtesting script
    # after results are generated
    
    visualizer = BacktestVisualizer()
    
    # Example: Load and visualize results
    try:
        # Look for the most recent results file
        import glob
        result_files = glob.glob("backtest_results_*.csv")
        
        if result_files:
            latest_file = max(result_files)
            print(f"Loading results from: {latest_file}")
            
            results_df = pd.read_csv(latest_file)
            
            # Create visualizations
            visualizer.create_performance_report(results_df)
            visualizer.plot_performance_zones(results_df)
            visualizer.plot_optimization_heatmap(results_df, 'composite_score')
            
        else:
            print("No backtest results found. Run backtesting_engine.py first.")
            
    except Exception as e:
        print(f"Error loading results: {e}")
        print("Make sure you have run the backtesting engine first.")

if __name__ == "__main__":
    main()
