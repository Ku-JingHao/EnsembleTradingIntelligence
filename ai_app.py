import plotly.graph_objects as go
import shinyswatch
import pandas as pd
import os
import shutil
from shiny import App, ui, render, reactive
from plotly.subplots import make_subplots
from datetime import datetime
from ai_stock_analysis import *
from functools import lru_cache
from ai_parameter_optimization import optimize_ai_enhanced_parameters

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
                "ai_model_select",
                "ML Model",
                choices=["Linear Regression", "Random Forest", "XGBoost", "LSTM", "GRU", "Ensemble"],
                selected="Linear Regression"
            ),
            ui.input_select(
                "stock_select",
                "Stock Name",
                choices=["TSLA", "GTLB", "HRMY", "JPM","YELP","AUPH"]
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
                choices=["1d", "1h"],  # REMOVED: "30m"
                selected="1h"
            ),
            
            ui.input_select(
                "strategy_select",
                "ML Enhanced Strategy",
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
                    "🚀 ML Analysis",
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
                ui.input_action_button(
                    "show_ai_info",
                    "🤖 ML Info",
                    class_="btn-info"
                ),
                style="display: flex; gap: 10px; margin-bottom: 15px; justify-content: flex-end;"
            ),
            ui.navset_card_tab(
                ui.nav_panel(
                    "ML Parameter Optimization",
                    ui.output_table("optimization_results"),
                    ui.tags.style("""
                        .dataframe th, .dataframe td {
                            text-align: center;
                        }
                    """)
                ),
                ui.nav_panel(
                    "ML Enhanced Charts",
                    ui.row(
                        ui.column(3,
                            ui.value_box(
                                "ML Model",
                                ui.output_text("ai_model_used"),
                                showcase=ui.HTML('<i class="fas fa-robot" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="info",
                                style="height: 100px;"
                            )
                        ),
                        ui.column(3,
                            ui.value_box(
                                "Trade Frequency",
                                ui.output_text("trade_freq"),
                                showcase=ui.HTML('<i class="fas fa-exchange-alt" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="primary",
                                style="height: 100px;"
                            )
                        ),
                        ui.column(2,
                            ui.value_box(
                                "ROI %",
                                ui.output_text("roi"),
                                showcase=ui.HTML('<i class="fas fa-percentage" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="success",
                                style="height: 100px;"
                            )
                        ),
                        ui.column(2,
                            ui.value_box(
                                "Max Profit",
                                ui.output_text("max_profit"),
                                showcase=ui.HTML('<i class="fas fa-arrow-up" style="font-size: 25px; margin-bottom: 10px;"></i>'),
                                theme="success",
                                style="height: 100px;"
                            )
                        ),
                        ui.column(2,
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
                                ui.card_header("🤖 ML-Enhanced Stock Price and Technical Indicator"),
                                ui.output_ui("combined_chart")
                            )
                        )
                    ),
                )
            )
        )
    ),
    title="🚀 ML-Enhanced Stock Trading System",
    theme=shinyswatch.theme.cosmo()
)

def server(input, output, session):
    # Reactive values for caching
    cached_data = reactive.Value({})
    trade_logs = reactive.Value(pd.DataFrame())
    active_state = reactive.Value(False)
    optimization_results_store = reactive.Value(None)
    optimization_chart_data_store = reactive.Value(None)  # ADD: Store the exact optimization chart data
    
    @reactive.Effect
    def _():
        """Clear cache when inputs change."""
        input.stock_select()
        input.ai_model_select()
        input.indicator_select()
        input.start_date()
        input.end_date()
        input.strategy_select()
        input.interval_select()
        cached_data.set({})
        optimization_chart_data_store.set(None)  # Clear optimization chart data

    @reactive.Effect
    @reactive.event(input.active_button)
    def _():
        active_state.set(True)
    
    @reactive.Effect
    @reactive.event(input.reset_button)
    def _():
        active_state.set(False)
        cached_data.set({})
        optimization_chart_data_store.set(None)  # Clear optimization chart data
        # Add notification for reset
        ui.notification_show(
            "🤖 ML Analysis Has Been Reset Successfully!",
            duration=3000,
            type="warning"
        )

    def get_filtered_data():
        """Get filtered data - now uses EXACT optimization data for consistency."""
        cache_key = 'filtered_data'
        
        # CHECK: If we have optimization chart data, use it directly for 100% consistency
        opt_chart_data = optimization_chart_data_store.get()
        if opt_chart_data is not None:
            print("🎯 Using EXACT optimization data for charts - guaranteed consistency!")
            
            # Extract all data from optimization results
            filtered_data = opt_chart_data['filtered_data']
            filtered_signals = opt_chart_data['signals']
            indicators = opt_chart_data['indicators']
            predicted_data = opt_chart_data['predicted_data']
            backtest_results = opt_chart_data['backtest_results']
            
            # Get best parameters from optimization results
            opt_results = optimization_results_store.get()
            best_params = {
                'Fast_Length': int(opt_results.iloc[0]['fast']),
                'Slow_Length': int(opt_results.iloc[0]['slow']),
                'Signal_Length': int(opt_results.iloc[0]['signal'])
            }
            
            print(f"🎯 OPTIMIZATION DATA USED FOR CHARTS:")
            print(f"   📊 Parameters: Fast={best_params['Fast_Length']}, Slow={best_params['Slow_Length']}, Signal={best_params['Signal_Length']}")
            print(f"   📈 ROI: {backtest_results['roi']:.6f}%")
            print(f"   🔄 Total Trades: {backtest_results['total_trades']}")
            print(f"   📊 Total Signals: {backtest_results['total_signals']}")
            print(f"   ✅ Confirmed Signals: {backtest_results['confirmed_signals']}")
            print(f"   📈 Final Value: ${backtest_results['final_value']:.2f}")

            print(f"📅 Total data points in user's selected date range: {len(filtered_data)}")
            
            # Count actual successful ML predictions (non-null, valid predictions)
            if predicted_data is not None and not predicted_data.empty:
                # Count non-null predictions in the Close column
                successful_predictions = predicted_data['Close'].notna().sum()
                print(f"🤖 Total ML predictions successfully generated: {successful_predictions}")
            else:
                print(f"🤖 Total ML predictions successfully generated: 0")

            # Set trade logs from optimization data
            trade_logs.set(backtest_results['trade_log'])
            
            # Cache the results using EXACT optimization data
            cached_results = {
                'filtered_data': filtered_data,
                'filtered_signals': filtered_signals,
                'stock_data': filtered_data,  # Use filtered data as stock data
                'best_params': best_params,
                'indicators': indicators,
                'predicted_data': predicted_data,
                'backtest_results': backtest_results
            }
            cached_data.set({cache_key: cached_results})
            
            return cached_results

        # FALLBACK: Use cached data if available
        if cache_key in cached_data.get():
            cached_result = cached_data.get()[cache_key]
            print("🔄 Using cached chart data")
            return cached_result
        
        # WARNING: No optimization data available
        print("⚠️ No optimization data available. Please run optimization first for consistent results.")
        empty_data = pd.DataFrame()
        empty_results = {
            'filtered_data': empty_data,
            'filtered_signals': empty_data,
            'stock_data': empty_data,
            'best_params': {'Fast_Length': 12, 'Slow_Length': 26, 'Signal_Length': 9},
            'indicators': empty_data,
            'predicted_data': empty_data,
            'backtest_results': {
                'final_value': 10000,
                'total_trades': 0,
                'total_signals': 0,
                'confirmed_signals': 0,
                'roi': 0,
                'max_profit': 0,
                'max_loss': 0,
                'trade_log': []
            }
        }
        cached_data.set({cache_key: empty_results})
        return empty_results

    @output
    @render.ui
    @reactive.event(input.active_button)
    def combined_chart():
        results = get_filtered_data()
        
        # Check if we have actual data
        if results['filtered_data'].empty:
            return ui.div(
                ui.tags.h4("📊 No Chart Data Available"),
                ui.tags.p("Please run ML optimization first to generate charts."),
                ui.tags.p("Click '🚀 ML Analysis' button to start optimization."),
                style="text-align: center; padding: 50px; color: #666;"
            )
        
        fig = create_ai_enhanced_chart(
            results['filtered_data'],
            input.indicator_select(),
            input.strategy_select(),
            results['filtered_signals'],
            results['indicators'],
            results['predicted_data'],
            input.ai_model_select()
        )
        return ui.HTML(fig.to_html(full_html=False))

    @output
    @render.table
    @reactive.event(input.active_button)
    def optimization_results():
        if not active_state.get():
            return pd.DataFrame()
        
        # Show loading indicator
        ui.notification_show(
            f"🤖 Running ML-Enhanced Parameter Optimization with {input.ai_model_select()}...", 
            duration=None, 
            type="default", 
            id="optimization_loading"
        )
            
        # 🔥 STEP 1: Calculate lag requirements with INTERVAL awareness
        from ai_stock_analysis import calculate_lag_requirements, get_extended_stock_data
        
        # Calculate required extension with interval information
        max_param = input.param_range_max()
        extension_needed = calculate_lag_requirements(
            input.ai_model_select(), 
            input.param_range_min(), 
            max_param, 
            max_param,
            interval=input.interval_select()  # 🔥 ADD: Pass interval
        )
        
        # 🔥 STEP 2: Fetch extended data with interval-appropriate extension
        try:
            extended_data = get_extended_stock_data(
                input.stock_select(),
                input.start_date(),
                input.end_date(),
                input.interval_select(),
                extension_days=extension_needed
            )
            
            print(f"🔄 Using extended dataset: {len(extended_data)} points (extension: {extension_needed} days for {input.interval_select()})")
            
        except Exception as e:
            print(f"⚠️ Extended data fetch failed: {e}, using standard data")
            extended_data = get_stock_data(input.stock_select(), input.interval_select())
        
        # Add data validation
        if len(extended_data) < 100:
            ui.notification_remove("optimization_loading")
            ui.notification_show(
                "❌ Insufficient data points for ML analysis. Please select a wider date range or different interval.",
                duration=5000,
                type="error"
            )
            return pd.DataFrame()
        
        try:
            # 🔥 STEP 3: Optimize parameters with extended data
            results, chart_data = optimize_ai_enhanced_parameters(
                extended_data,  # Use extended data instead of standard data
                input.indicator_select(),
                input.strategy_select(),
                input.ai_model_select(),
                input.start_date(),
                input.end_date(),
                param_range=(input.param_range_min(), input.param_range_max()),
                transaction_fee=float(input.transaction_fee().strip('%'))/100,
                interval=input.interval_select(),  # 🔥 ADD: Pass interval
                stock_name=input.stock_select()    # 🔥 ADD: Pass stock_name
            )
            
            # Hide loading indicator
            ui.notification_remove("optimization_loading")
            
            if results.empty:
                optimization_results_store.set(None)
                optimization_chart_data_store.set(None)
                ui.notification_show(
                    f"🤖 {input.ai_model_select()} analysis completed but no optimal parameters found.",
                    duration=5000,
                    type="warning"
                )
                return None
            
            optimization_results_store.set(results)
            optimization_chart_data_store.set(chart_data)
            
            # Clear cache to force refresh with optimization data
            cached_data.set({})
            print("🗑️ Cache cleared - charts will now use EXACT optimization data")
            
            ui.notification_show(
                f"✅ ML-Enhanced optimization with {input.ai_model_select()} completed successfully!",
                duration=3000,
                type="success"
            )
            return results
            
        except Exception as e:
            # Hide loading indicator and show error message
            ui.notification_remove("optimization_loading")
            ui.notification_show(f"❌ Error during ML optimization: {str(e)}", 
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
    @reactive.event(input.active_button)
    def ai_model_used():
        if not active_state.get():
            return "Not Active"
        
        # Get the parameters being used from optimization
        results = get_filtered_data()
        params = results['best_params']
        
        # Show special name for ensemble
        model_name = input.ai_model_select()
        if model_name == "Ensemble":
            display_name = "LR + LSTM"
        else:
            display_name = model_name
        
        return f"{display_name} (F:{params['Fast_Length']}, S:{params['Slow_Length']}, Sig:{params['Signal_Length']})"

    @output
    @render.text
    @reactive.event(input.active_button)
    def trade_freq():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        confirmed = results['backtest_results']['confirmed_signals']
        total = results['backtest_results']['total_signals']
        return f"{results['backtest_results']['total_trades']} trades ({confirmed}/{total})"

    @output
    @render.text
    @reactive.event(input.active_button)
    def roi():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        return f"{results['backtest_results']['roi']:.2f}%"

    @output
    @render.text
    @reactive.event(input.active_button)
    def max_profit():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        return f"${results['backtest_results']['max_profit']:.2f}"

    @output
    @render.text
    @reactive.event(input.active_button)
    def max_loss():
        if not active_state.get():
            return "N/A"
        results = get_filtered_data()
        return f"${results['backtest_results']['max_loss']:.2f}"

    @reactive.effect
    @reactive.event(input.show)
    def show_important_message():
        # Modal Content with ML-enhanced message
        message = ui.modal(
            ui.tags.div(
                ui.tags.h3("🤖 ML-Enhanced Trading Strategies"),
                ui.tags.p("Your selected ML model validates each signal before execution:"),
                ui.tags.hr(),
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
                    ui.tags.h4("🤖 ML Enhancement Benefits:"),
                    ui.tags.ul(
                        ui.tags.li("• Reduces false signals by using ML predictions"),
                        ui.tags.li("• Improves trading accuracy with machine learning"),
                        ui.tags.li("• Combines technical analysis with ML forecasting"),
                        ui.tags.li("• Shows both confirmed and rejected signals in results")
                    )
                ),
                ui.tags.hr(),
                ui.tags.div(
                    ui.tags.h4("Available ML Models:"),
                    ui.tags.ul(
                        ui.tags.li("🔹 Linear Regression: Fast, simple predictions"),
                        ui.tags.li("🌳 Random Forest: Ensemble learning approach"),
                        ui.tags.li("⚡ XGBoost: Advanced gradient boosting"),
                        ui.tags.li("🧠 LSTM: Deep learning for time series"),
                        ui.tags.li("🔄 GRU: Efficient recurrent neural network"),
                        ui.tags.li("�� Ensemble: Combines Linear Regression and LSTM for balanced predictions")
                    )
                )
            ),
            easy_close=True,
            footer=None,
            size="large"
        )
        ui.modal_show(message)

    @reactive.effect
    @reactive.event(input.show_ai_info)
    def show_ai_model_info():
        """Show information about the selected ML model"""
        try:
            from model_loader import ModelLoader
            loader = ModelLoader()
            info = loader.get_model_info()
            
            model_descriptions = {
                "Linear Regression": "Fast and interpretable model good for linear relationships",
                "Random Forest": "Ensemble model that combines multiple decision trees",
                "XGBoost": "Advanced gradient boosting with high accuracy",
                "LSTM": "Deep learning model that captures long-term dependencies",
                "GRU": "Efficient recurrent neural network for sequence modeling",
                "Ensemble": "Combines Linear Regression and LSTM for balanced predictions"
            }
            
            selected_model = input.ai_model_select()
            description = model_descriptions.get(selected_model, "Advanced ML model")
            
            message = ui.modal(
                ui.tags.div(
                    ui.tags.h3(f"🤖 {selected_model} Model Information"),
                    ui.tags.hr(),
                    ui.tags.h5("Model Description:"),
                    ui.tags.p(description),
                    ui.tags.h5("Deployment Status:"),
                    ui.tags.p(f"Status: {info['status']}" if info else "Model information unavailable"),
                    ui.tags.h5("How It Works:"),
                    ui.tags.ul(
                        ui.tags.li("1. Analyzes historical stock price patterns"),
                        ui.tags.li("2. Generates one-step-ahead price predictions"),
                        ui.tags.li("3. Validates MACD signals using predictions"),
                        ui.tags.li("4. Only executes trades when ML confirms the signal")
                    ),
                    ui.tags.hr(),
                    ui.tags.div(
                        ui.tags.h5("Available Models:"),
                        *[
                            ui.tags.p(f"{'✅' if model['name'] == selected_model.lower().replace(' ', '_') else '🔹'} {model['display_name']} ({model['type']})")
                            for model in info['available_models']
                        ] if info and 'available_models' in info else [ui.tags.p("Model information loading...")]
                    )
                ),
                easy_close=True,
                footer=None,
                size="medium"
            )
            ui.modal_show(message)
            
        except Exception as e:
            # Fallback message if model info can't be loaded
            message = ui.modal(
                ui.tags.div(
                    ui.tags.h3(f"🤖 {input.ai_model_select()} Model"),
                    ui.tags.p("This ML model enhances traditional MACD signals by:"),
                    ui.tags.ul(
                        ui.tags.li("• Predicting next-step price movements"),
                        ui.tags.li("• Validating buy/sell signals"),
                        ui.tags.li("• Reducing false signals"),
                        ui.tags.li("• Improving overall trading accuracy")
                    ),
                    ui.tags.p(f"Note: {str(e)}")
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
                    ui.tags.h4("No ML Trade Logs Available"),
                    ui.tags.p("Please run an ML analysis first.")
                ),
                easy_close=True,
                footer=None
            )
            ui.modal_show(message)
            return

        # Create a modal with enhanced trade logs
        if isinstance(logs, pd.DataFrame):
            trade_log_content = ui.tags.div(
                ui.tags.h4("🤖 ML-Enhanced Trade Log"),
                ui.tags.p("Shows both confirmed (✅) and rejected (❌) signals"),
                *[
                    ui.tags.div(
                        ui.tags.p(
                            f"{'✅' if log.get('validation') == 'CONFIRMED' else '❌'} "
                            f"{log['type']} | Date: {log['date']}, "
                            f"Price: ${log['price']:.2f}, "
                            f"Predicted: ${log.get('predicted_next', 0):.2f}, "
                            f"Shares: {log['shares']}, "
                            f"Status: {log.get('validation', 'N/A')}"
                        ),
                        style=f"margin-bottom: 10px; border-bottom: 1px solid #eee; "
                              f"color: {'green' if log.get('validation') == 'CONFIRMED' else 'red'};"
                    )
                    for log in logs.to_dict('records')
                ]
            )
        else:
            trade_log_content = ui.tags.div(
                ui.tags.h4("🤖 ML-Enhanced Trade Log"),
                ui.tags.p("Shows both confirmed (✅) and rejected (❌) signals"),
                *[
                    ui.tags.div(
                        ui.tags.p(
                            f"{'✅' if log.get('validation') == 'CONFIRMED' else '❌'} "
                            f"{log['type']} | Date: {log['date']}, "
                            f"Price: ${log['price']:.2f}, "
                            f"Predicted: ${log.get('predicted_next', 0):.2f}, "
                            f"Shares: {log['shares']}, "
                            f"Status: {log.get('validation', 'N/A')}"
                        ),
                        style=f"margin-bottom: 10px; border-bottom: 1px solid #eee; "
                              f"color: {'green' if log.get('validation') == 'CONFIRMED' else 'red'};"
                    )
                    for log in logs
                ]
            )

        message = ui.modal(
            trade_log_content,
            title="🤖 ML-Enhanced Detailed Trade Log",
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