import plotly.graph_objects as go
import shinyswatch
import pandas as pd
import os
import shutil
from shiny import App, ui, render, reactive
from plotly.subplots import make_subplots
from datetime import datetime
from stock_analysis import *
from functools import lru_cache
from parameter_optimization import optimize_parameters


def cleanup_upload_dir(dir_path):
    """Safely cleanup upload directory"""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
    except Exception:
        pass  # Ignore cleanup errors

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.link(
            rel="stylesheet",
            href="https://use.fontawesome.com/releases/v5.15.4/css/all.css",
            integrity="sha384-DyZ88mC6Up2uqS4h/KRgHuoeGwBcD4Ng9SiP4dIRy0EXTlnuz47vAwmeGwVChigm",
            crossorigin="anonymous"
        )
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_select(
                "stock_select",
                "Stock Name",
                choices=["TSLA", "NVDA", "JPM", "GTLB", "PRI", "FL", "HRMY", "YELP", "AUPH"]
            ),
            ui.input_select(
                "indicator_select", 
                "Technical Indicator",
                choices=["MACD", "Zero Lag MACD"],
                selected="Zero Lag MACD"
            ),
            ui.input_select(
                "transaction_fee",
                "Transaction Fee",
                choices=["1%", "2%", "3%", "4%", "5%"],
                selected="1%"
            ),
            ui.div(
                ui.input_date(
                    "start_date",
                    "Start Date",
                    value="2024-06-01",
                    min="2023-01-01",
                    max=datetime.now()
                ),
                ui.input_date(
                    "end_date",
                    "End Date",
                    value="2024-07-31",
                    min="2023-01-01",
                    max=datetime.now()
                ),
                style="display: flex; height:65px; gap: 10px;"  # Flexbox for side-by-side layout
            ),
            ui.input_select(
                "interval_select",
                "Interval",
                choices=["1d", "1h", "30m"],
                selected="1h"
            ),
            
            ui.input_select(
                "strategy_select",
                "Strategy",
                choices=["Buy Above Sell Above", "Buy Below Sell Above", "Buy Above Sell Below", "Buy Below Sell Below", "Histogram Trend Reversal"]
            ),
            ui.div(
                ui.tags.h6("Parameter Range"),  # Title for the section
                ui.div(
                    ui.input_numeric("param_range_min", "", value=5, min=1),
                    ui.input_numeric("param_range_max", "", value=30, min=1),
                    style="display: flex; gap: 10px;"  # Flexbox for side-by-side layout
                ),
                style=" height:65px;"  # Optional style to add spacing between sections
            ),
            ui.div(
                ui.input_action_button(
                    "active_button",
                    "Analysis",
                    class_="btn-success",  # Green color for Analysis button
                    style="margin: 10px 10px 10px 0; width: 150px; border-radius: 10px;"  # Same width and rounded corners
                ),
                ui.input_action_button(
                    "reset_button",
                    "Reset",
                    class_="btn-danger",  # Red color for Reset button
                    style="margin: 10px 0; width: 150px; border-radius: 10px;"  # Same width and rounded corners
                ),
                style="display: flex; gap: 10px;"  # Flexbox for side-by-side layout
            )
        ),
        ui.div(
            ui.div(
                ui.input_action_button(
                    "show",
                    "Instructions",
                    class_="btn-secondary"
                ),
                ui.input_action_button(
                    "show_trades",
                    "Trade Log",
                    class_="btn-secondary"
                ),
                style="display: flex; gap: 10px; margin-bottom: 15px; justify-content: flex-end;"
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "Parameter Optimization",
                    ui.output_table("optimization_results"),
                    ui.tags.style("""
                        .dataframe th, .dataframe td {
                            text-align: center;
                        }
                    """)
                ),
                ui.nav_panel(
                    "Charts",
                    ui.row(
                        ui.column(3,
                            ui.value_box(
                                "Trade Frequency",
                                ui.output_text("trade_freq"),
                                showcase=ui.HTML('<i class="fas fa-exchange-alt" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="primary",
                                style="height: 100px;"
                            )
                        ),
                        ui.column(3,
                            ui.value_box(
                                "ROI %",
                                ui.output_text("roi"),
                                showcase=ui.HTML('<i class="fas fa-percentage" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="success",
                                style="height: 100px;"
                            )
                        ),
                        ui.column(3,
                            ui.value_box(
                                "Max Profit",
                                ui.output_text("max_profit"),
                                showcase=ui.HTML('<i class="fas fa-arrow-up" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="info",
                                style="height: 100px;"
                            )
                        ),
                        ui.column(3,
                            ui.value_box(
                                "Max Loss",
                                ui.output_text("max_loss"),
                                showcase=ui.HTML('<i class="fas fa-arrow-down" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="warning",
                                style="height: 100px;"
                            )
                        )
                    ),
                    ui.row(
                        ui.column(12,
                            ui.card(
                                ui.card_header("Stock Price and Technical Indicator"),
                                ui.output_ui("combined_chart")
                            )
                        )
                    ),
                )
            )
        )
    ),
    title="Stock Technical Analysis",
    theme=shinyswatch.theme.cosmo()
)
def server(input, output, session):
    # Reactive values for caching
    cached_data = reactive.Value({})
    trade_logs = reactive.Value(pd.DataFrame())
    active_state = reactive.Value(False)
    optimization_results_store = reactive.Value(None)
    
    @reactive.Effect
    def _():
        """Clear cache when inputs change."""
        input.stock_select()
        input.indicator_select()
        input.start_date()
        input.end_date()
        input.strategy_select()
        input.interval_select()
        cached_data.set({})

    @reactive.Effect
    @reactive.event(input.active_button)
    def _():
        active_state.set(True)
    
    @reactive.Effect
    @reactive.event(input.reset_button)
    def _():
        active_state.set(False)
        cached_data.set({})
        # Add notification for reset
        ui.notification_show(
            "Analysis Has Been Reset Successfully!",
            duration=3000,
            type="warning"
        )

    @reactive.Effect
    def _():
        """Show warning for 30-minute interval selection"""
        if input.interval_select() == "30m":
            ui.notification_show(
                "For 30-minute intervals, please select a start date and end date within a 20-day range for best results.",
                duration=5000,
                type="warning"
            )

    def get_filtered_data():
        """Get filtered data with caching."""
        cache_key = 'filtered_data'
        if cache_key in cached_data.get():
            return cached_data.get()[cache_key]

        # Get stock data
        stock_data = get_stock_data(input.stock_select(), input.interval_select())
        
        # Get best parameters from optimization results
        opt_results = optimization_results_store.get()
        if opt_results is not None and not opt_results.empty:
            best_params = {
                'Fast_Length': int(opt_results.iloc[0]['Fast_Length']),
                'Slow_Length': int(opt_results.iloc[0]['Slow_Length']),
                'Signal_Length': int(opt_results.iloc[0]['Signal_Length'])
            }
        else:
            best_params = {
                'Fast_Length': 12,
                'Slow_Length': 26,
                'Signal_Length': 9
            }
        
        # Calculate indicators once
        if input.indicator_select() == "MACD":
            indicators = calculate_macd(
                stock_data,
                fast_length=best_params['Fast_Length'],
                slow_length=best_params['Slow_Length'],
                signal_length=best_params['Signal_Length']
            )
        else:
            indicators = calculate_zero_lag_macd(
                stock_data,
                fast_length=best_params['Fast_Length'],
                slow_length=best_params['Slow_Length'],
                signal_length=best_params['Signal_Length']
            )
        
        # Generate signals once
        signals = generate_signals(
            stock_data, 
            input.indicator_select(), 
            input.strategy_select(),
            fast_length=best_params['Fast_Length'],
            slow_length=best_params['Slow_Length'],
            signal_length=best_params['Signal_Length']
        )

        start_date = pd.to_datetime(input.start_date()).tz_localize('America/New_York')
        end_date = pd.to_datetime(input.end_date()).tz_localize('America/New_York')

        filtered_data = stock_data.loc[start_date:end_date]
        filtered_signals = signals.loc[start_date:end_date]
        filtered_indicators = indicators.loc[start_date:end_date]
        
        # Perform backtest once
        backtest_results = backtest_strategy(filtered_data, filtered_signals, transaction_fee=float(input.transaction_fee().strip('%'))/100)
        trade_logs.set(backtest_results['trade_log'])

        # Cache all results
        cached_results = {
            'filtered_data': filtered_data,
            'filtered_signals': filtered_signals,
            'stock_data': stock_data,
            'best_params': best_params,
            'indicators': filtered_indicators,
            'backtest_results': backtest_results
        }
        cached_data.set({cache_key: cached_results})
        
        return cached_results

    @output
    @render.ui
    @reactive.event(input.active_button)  # Only update when active button is clicked
    def combined_chart():
        results = get_filtered_data()
        fig = create_combined_chart(
            results['filtered_data'],
            input.indicator_select(),
            input.strategy_select(),
            results['filtered_signals'],
            results['indicators'],
            results['indicators']  # Use the same indicators for both MACD types
        )
        return ui.HTML(fig.to_html(full_html=False))

    @output
    @render.table
    @reactive.event(input.active_button)  # Only update when active button is clicked
    def optimization_results():
        if not active_state.get():
            return pd.DataFrame()  # Return empty dataframe if not active
        
        # Show loading indicator
        ui.notification_show(
            "Running Parameter Optimization...", 
            duration=None, 
            type="default", 
            id="optimization_loading"
        )
            
        # Get stock data
        data = get_stock_data(input.stock_select(), input.interval_select())
        
        # Add data validation
        if len(data) < 100:  # Minimum required data points
            ui.notification_remove("optimization_loading")
            ui.notification_show(
                "Insufficient data points for analysis. Please select a wider date range or different interval.",
                duration=5000,
                type="error"
            )
            return pd.DataFrame()
        
        try:
            # Optimize parameters with date range
            results = optimize_parameters(
                data,
                input.indicator_select(),
                input.strategy_select(),
                input.start_date(),
                input.end_date(),
                param_range=(input.param_range_min(), input.param_range_max()),
                transaction_fee=float(input.transaction_fee().strip('%'))/100
            )
            
            # Hide loading indicator
            ui.notification_remove("optimization_loading")
            
            if results.empty:
                optimization_results_store.set(None)
                return None
            
            optimization_results_store.set(results)  # Store the results
            return results
            
        except Exception as e:
            # Hide loading indicator and show error message
            ui.notification_remove("optimization_loading")
            ui.notification_show(f"Error during optimization: {str(e)}", 
                               duration=5000, 
                               type="error")
            return pd.DataFrame()

    # Add custom CSS for notifications
    ui.insert_ui(
        ui.tags.style("""
            .shiny-notification {
                font-size: 30px !important;
                padding: 15px 25px !important;
                border-radius: 8px !important;
                border-left: 5px solid #007bff !important;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
                margin: 20px !important;
                width: 600px !important;
                position: fixed !important;
                right: 20px !important;
                bottom: 20px !important;
            }
            
            #shiny-notification-panel {
                position: fixed !important;
                bottom: 0 !important;
                right: 0 !important;
                width: 100px !important;
                z-index: 99999 !important;
            }
        """),
        "head"
    )

    @output
    @render.text
    @reactive.event(input.active_button)  # Only update when active button is clicked
    def trade_freq():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        return f"{results['backtest_results']['total_trades']} trades"

    @output
    @render.text
    @reactive.event(input.active_button)  # Only update when active button is clicked
    def roi():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        return f"{results['backtest_results']['roi']:.2f}%"

    @output
    @render.text
    @reactive.event(input.active_button)  # Only update when active button is clicked
    def max_profit():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        return f"${results['backtest_results']['max_profit']:.2f}"

    @output
    @render.text
    @reactive.event(input.active_button)  # Only update when active button is clicked
    def max_loss():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        return f"${results['backtest_results']['max_loss']:.2f}"

    @reactive.effect
    @reactive.event(input.show)
    def show_important_message():
        # Modal Content with styled message
        message = ui.modal(
            ui.tags.div(
                ui.tags.h4("Buy Above Sell Above:"),
                ui.tags.h5("Buy Signal:"),
                ui.tags.ul(
                    ui.tags.li("1) The Zero Lag MACD crosses above the Signal Line."),
                    ui.tags.li("2) Crossover point is above the zero axis."),
                ),
                ui.tags.h5("Sell Signal:"),
                ui.tags.ul(
                    ui.tags.li("1) The Zero Lag MACD crosses below the Signal Line."),
                    ui.tags.li("2) Crossover point is above the zero axis."),
                ),
                ui.tags.hr(),
                ui.tags.div(
                    ui.tags.h4("Buy Below Sell Above:"),
                    ui.tags.h5("Buy Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) The Zero Lag MACD crosses above the Signal Line."),
                        ui.tags.li("2) Crossover point is below the zero axis."),
                    ),
                    ui.tags.h5("Sell Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) The Zero Lag MACD crosses below the Signal Line."),
                        ui.tags.li("2) Crossover point is above the zero axis."),
                    )
                ),
                ui.tags.hr(),
                ui.tags.div(
                    ui.tags.h4("Buy Above Sell Below:"),
                    ui.tags.h5("Buy Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) The Zero Lag MACD crosses above the Signal Line."),
                        ui.tags.li("2) Crossover point is below the zero axis."),
                    ),
                    ui.tags.h5("Sell Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) The Zero Lag MACD crosses below the Signal Line."),
                        ui.tags.li("2) Crossover point is below the zero axis."),
                    )
                ),
                ui.tags.hr(),
                ui.tags.div(
                    ui.tags.h4("Buy Below Sell Below:"),
                    ui.tags.h5("Buy Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) The Zero Lag MACD crosses above the Signal Line."),
                        ui.tags.li("2) Crossover point is below the zero axis."),
                    ),
                    ui.tags.h5("Sell Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) The Zero Lag MACD crosses below the Signal Line."),
                        ui.tags.li("2) Crossover point is below the zero axis."),
                    )
                ),
                ui.tags.hr(),
                ui.tags.div(
                    ui.tags.h4("Histogram Trend Reversal:"),
                    ui.tags.h5("Buy Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) Look for negative histogram values that are continuously decreasing."),
                        ui.tags.li("2) Once they reverse (start increasing after a low point), trigger a buy signal.")
                    ),
                    ui.tags.h5("Sell Signal:"),
                    ui.tags.ul(
                        ui.tags.li("1) Look for positive histogram values that are continuously increasing."),
                        ui.tags.li("2) Once they reverse (start decreasing after a high point), trigger a sell signal.")
                    )
                ),
                ui.tags.hr(),
                ui.tags.div(
                    ui.tags.h4("Limitations:"),
                    ui.tags.ul(
                        ui.tags.li("Please ensure that every time you click 'Analysis' or 'Reset' to 'Analysis' for new inputs. You should remain on the parameter optimization page until the results are displayed. Only then click the 'Chart' tab to view the chart and backtest results."),
                        ui.tags.li("Please note: To test the 30-minute interval, we recommend selecting a start date and end date within a 20-day range."),
                    )
                ),
            ),
            easy_close=True,
            footer=None
        )
        ui.modal_show(message)

    @reactive.effect
    @reactive.event(input.show_trades)  # Listen to the correct button
    def show_trade_log():
        # Get the trade logs
        logs = trade_logs.get()
        
        # If logs is empty or None
        if logs is None or (isinstance(logs, pd.DataFrame) and logs.empty) or (isinstance(logs, list) and not logs):
            message = ui.modal(
                ui.tags.div(
                    ui.tags.h4("No Trade Logs Available"),
                    ui.tags.p("Please run a backtest first.")
                ),
                easy_close=True,
                footer=None
            )
            ui.modal_show(message)
            return

        # Create a modal with trade logs
        if isinstance(logs, pd.DataFrame):
            trade_log_content = ui.tags.div(
                ui.tags.h4("Trade Log"),
                *[
                    ui.tags.div(
                        ui.tags.p(
                            f"{log['type']} | Date: {log['date']}, "
                            f"Price: ${log['price']:.2f}, "
                            f"Shares: {log['shares']}, "
                            f"Cash Left: ${log['cash_left']:.2f}"
                        ),
                        style="margin-bottom: 10px; border-bottom: 1px solid #eee;"
                    )
                    for log in logs.to_dict('records')
                ]
            )
        else:
            trade_log_content = ui.tags.div(
                ui.tags.h4("Trade Log"),
                *[
                    ui.tags.div(
                        ui.tags.p(
                            f"{log['type']} | Date: {log['date']}, "
                            f"Price: ${log['price']:.2f}, "
                            f"Shares: {log['shares']}, "
                            f"Cash Left: ${log['cash_left']:.2f}"
                        ),
                        style="margin-bottom: 10px; border-bottom: 1px solid #eee;"
                    )
                    for log in logs
                ]
            )

        message = ui.modal(
            trade_log_content,
            title="Detailed Trade Log",
            easy_close=True,
            footer=None,
            size="large"
        )
        ui.modal_show(message)

    # Add cleanup handler for session end
    @session.on_ended
    def _():
        if hasattr(session, '_fileupload_basedir'):
            cleanup_upload_dir(session._fileupload_basedir)

app = App(app_ui, server)
