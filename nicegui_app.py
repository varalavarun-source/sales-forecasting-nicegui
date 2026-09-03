
from nicegui import ui
import pandas as pd
import numpy as np
import joblib
import traceback
from pathlib import Path


# ============================================================
# FILE PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "sales_forecasting_model.pkl"
DATA_PATH = BASE_DIR / "Sales_Forcasting_Dataset.xlsx"


# ============================================================
# LOAD MODEL
# ============================================================

bundle = joblib.load(MODEL_PATH)

model = bundle["model"]
model_features = bundle["model_features"]
categorical_features = bundle["categorical_features"]
category_levels = bundle["category_levels"]


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_excel(DATA_PATH)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Product_ID", "Store_ID", "Date"]
).reset_index(drop=True)


# ============================================================
# PRODUCT OPTIONS
# ============================================================

product_info = (
    df[
        [
            "Product_ID",
            "Product_Name"
        ]
    ]
    .drop_duplicates()
)

product_options = []

product_map = {}

for _, row in product_info.iterrows():

    label = (
        f"{row['Product_ID']} - "
        f"{row['Product_Name']}"
    )

    product_options.append(label)

    product_map[label] = row["Product_ID"]


# ============================================================
# STORE OPTIONS
# ============================================================

store_info = (
    df[
        [
            "Store_ID",
            "Store_Location"
        ]
    ]
    .drop_duplicates()
)

store_options = []

store_map = {}

for _, row in store_info.iterrows():

    label = (
        f"{row['Store_ID']} - "
        f"{row['Store_Location']}"
    )

    store_options.append(label)

    store_map[label] = row["Store_ID"]


# ============================================================
# REGION OPTIONS
# ============================================================

region_options = sorted(
    df["Store_Location"]
    .dropna()
    .unique()
    .tolist()
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_season(month):

    if month in [12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5]:
        return "Spring"

    if month in [6, 7, 8]:
        return "Summer"

    return "Autumn"


def safe_category(column, value):

    levels = category_levels[column]

    if value in levels:
        return value

    return levels[0]


# ============================================================
# CREATE MODEL INPUT
# ============================================================

def create_model_row(
    future_date,
    latest,
    history,
    overrides=None
):

    if overrides is None:
        overrides = {}


    def lag(n):

        if len(history) >= n:
            return float(history[-n])

        return float(np.mean(history))


    def rolling(n):

        values = history[-n:]

        if len(values) == 0:
            return 0.0

        return float(np.mean(values))


    row = {

        "Product_ID":
            latest["Product_ID"],

        "Product_Name":
            latest["Product_Name"],

        "Category":
            latest["Category"],

        "Store_ID":
            latest["Store_ID"],

        "Store_Location":
            latest["Store_Location"],


        "Price":
            overrides.get(
                "Price",
                latest["Price"]
            ),

        "Discount_Percentage":
            overrides.get(
                "Discount_Percentage",
                latest["Discount_Percentage"]
            ),

        "Promotion_Flag":
            overrides.get(
                "Promotion_Flag",
                latest["Promotion_Flag"]
            ),

        "Stock_Availability":
            overrides.get(
                "Stock_Availability",
                latest["Stock_Availability"]
            ),


        "Day_of_Week":
            future_date.day_name(),

        "Month":
            future_date.month_name(),

        "Quarter":
            f"Q{future_date.quarter}",

        "Holiday_Flag":
            overrides.get(
                "Holiday_Flag",
                0
            ),

        "Is_Weekend":
            int(
                future_date.dayofweek in [5, 6]
            ),

        "Season":
            get_season(
                future_date.month
            ),

        "Weather":
            latest["Weather"],

        "Local_Event_Flag":
            overrides.get(
                "Local_Event_Flag",
                0
            ),


        "Competitor_Price":
            overrides.get(
                "Competitor_Price",
                latest["Competitor_Price"]
            ),

        "Economic_Indicator":
            overrides.get(
                "Economic_Indicator",
                latest["Economic_Indicator"]
            ),

        "Sales_Channel":
            latest["Sales_Channel"],

        "Customer_Segment":
            latest["Customer_Segment"],

        "Marketing_Spend":
            overrides.get(
                "Marketing_Spend",
                latest["Marketing_Spend"]
            ),


        "Lag_1":
            lag(1),

        "Lag_7":
            lag(7),

        "Lag_14":
            lag(14),

        "Lag_28":
            lag(28),

        "Rolling_Mean_7":
            rolling(7),

        "Rolling_Mean_14":
            rolling(14),

        "Rolling_Mean_28":
            rolling(28)
    }


    input_data = pd.DataFrame([row])


    # Restore exact categories used during training
    for column in categorical_features:

        value = safe_category(
            column,
            input_data.loc[0, column]
        )

        input_data[column] = pd.Categorical(
            [value],
            categories=category_levels[column]
        )


    # Exact feature order used during training
    input_data = input_data[
        model_features
    ]


    return input_data


# ============================================================
# FORECAST PRODUCT / STORE
# ============================================================

def forecast_product_store(
    product,
    store,
    start_date,
    days,
    overrides=None
):

    # Only use information available BEFORE forecast date
    history_df = df[
        (df["Product_ID"] == product)
        &
        (df["Store_ID"] == store)
        &
        (df["Date"] <= start_date)
    ].copy()


    history_df = history_df.sort_values(
        "Date"
    )


    # Need historical observations for lag 28
    if len(history_df) < 28:

        return None, None


    latest = history_df.iloc[-1]


    history = (
        history_df["Units_Sold"]
        .astype(float)
        .tolist()
    )


    results = []


    for day_number in range(
        1,
        days + 1
    ):

        future_date = (
            start_date
            +
            pd.Timedelta(
                days=day_number
            )
        )


        model_input = create_model_row(
            future_date,
            latest,
            history,
            overrides
        )


        prediction = float(
            model.predict(
                model_input
            )[0]
        )


        prediction = max(
            0,
            prediction
        )


        # Recursive forecasting:
        # future prediction becomes part of history
        history.append(prediction)


        results.append({

            "Date":
                future_date,

            "Predicted":
                prediction

        })


    return (
        pd.DataFrame(results),
        history_df
    )


# ============================================================
# LIGHT THEME
# ============================================================

ui.add_css("""

body {

    margin: 0;

    background: #f5f7fb;

    color: #111827;

    font-family:
        "Segoe UI",
        Arial,
        sans-serif;
}


.sidebar {

    background:
        linear-gradient(
            180deg,
            #eef4ff,
            #f7faff
        );

    border-right:
        1px solid #d9e3f0;

    min-height: 100vh;

    padding: 25px;
}


.main-area {

    background: #f5f7fb;

    min-height: 100vh;

    padding: 32px;
}


.main-title {

    color: #111827;

    font-size: 34px;

    font-weight: 800;
}


.main-subtitle {

    color: #64748b;

    font-size: 15px;

    margin-bottom: 25px;
}


.card-box {

    background: #ffffff;

    border:
        1px solid #e2e8f0;

    border-radius: 16px;

    padding: 22px;

    box-shadow:
        0 4px 15px
        rgba(15,23,42,.05);
}


.kpi-title {

    color: #64748b;

    font-size: 12px;

    font-weight: 700;
}


.kpi-value {

    color: #2563eb;

    font-size: 29px;

    font-weight: 800;

    margin-top: 6px;
}


.section-title {

    color: #111827;

    font-size: 24px;

    font-weight: 800;
}


.insight-box {

    width: 100%;

    background: #f8fbff;

    border:
        1px solid #dbeafe;

    border-radius: 10px;

    padding: 14px 16px;

    color: #334155;
}

""")


# ============================================================
# SIDEBAR
# ============================================================

with ui.row().classes(
    "w-full items-start gap-0 no-wrap"
):

    with ui.column().classes(
        "sidebar w-80"
    ):

        ui.label(
            "📈 Sales Intelligence"
        ).classes(
            "text-2xl font-bold text-blue-700"
        )


        ui.label(
            "XGBoost Forecasting Dashboard"
        ).classes(
            "text-gray-500 mb-5"
        )


        ui.separator()


        ui.label(
            "📦 Product Forecast"
        ).classes(
            "text-lg font-bold mt-4"
        )


        product_select = ui.select(
            product_options,
            value=product_options[0],
            label="Product"
        ).classes(
            "w-full"
        )


        store_select = ui.select(
            store_options,
            value=store_options[0],
            label="Store"
        ).classes(
            "w-full"
        )


        forecast_date = ui.input(
            label="Forecast Start Date",
            value="2026-09-01"
        ).classes(
            "w-full"
        )


        ui.label(
            "Use format YYYY-MM-DD"
        ).classes(
            "text-xs text-gray-500"
        )


        price = ui.number(
            label="Price",
            value=2500,
            min=0
        ).classes(
            "w-full"
        )


        discount = ui.number(
            label="Discount %",
            value=5,
            min=0,
            max=100
        ).classes(
            "w-full"
        )


        marketing = ui.number(
            label="Marketing Spend",
            value=3000,
            min=0
        ).classes(
            "w-full"
        )


        competitor_price = ui.number(
            label="Competitor Price",
            value=2500,
            min=0
        ).classes(
            "w-full"
        )


        economic_indicator = ui.number(
            label="Economic Indicator",
            value=103
        ).classes(
            "w-full"
        )


        promotion = ui.select(
            ["No", "Yes"],
            value="No",
            label="Promotion"
        ).classes(
            "w-full"
        )


        stock = ui.select(
            [
                "Available",
                "Out of Stock"
            ],
            value="Available",
            label="Stock Availability"
        ).classes(
            "w-full"
        )


        holiday = ui.select(
            ["No", "Yes"],
            value="No",
            label="Holiday"
        ).classes(
            "w-full"
        )


        local_event = ui.select(
            ["No", "Yes"],
            value="No",
            label="Local Event"
        ).classes(
            "w-full"
        )


        horizon = ui.number(
            label="Forecast Horizon (Days)",
            value=30,
            min=7,
            max=60
        ).classes(
            "w-full"
        )


        generate_button = ui.button(
            "🚀 GENERATE FORECAST"
        ).classes(
            "w-full mt-3"
        )


        ui.separator().classes(
            "my-6"
        )


        ui.label(
            "📍 Regional Analysis"
        ).classes(
            "text-lg font-bold"
        )


        region_select = ui.select(
            region_options,
            value=region_options[0],
            label="Region"
        ).classes(
            "w-full"
        )


        ranking_button = ui.button(
            "🏆 GENERATE PRODUCT RANKING"
        ).classes(
            "w-full mt-3"
        )


# ============================================================
# MAIN DASHBOARD
# ============================================================

    with ui.column().classes(
        "main-area flex-1"
    ):


        ui.label(
            "Sales Forecast Intelligence Dashboard"
        ).classes(
            "main-title"
        )


        ui.label(
            "Advanced demand forecasting, scenario modelling "
            "and regional product intelligence"
        ).classes(
            "main-subtitle"
        )


        status = ui.label(
            "Select your settings and generate a forecast."
        ).classes(
            "text-blue-600 font-medium mb-4"
        )


        # ====================================================
        # KPI CARDS
        # ====================================================

        with ui.row().classes(
            "w-full gap-4"
        ):


            with ui.column().classes(
                "card-box flex-1"
            ):

                ui.label(
                    "TOTAL FORECAST"
                ).classes(
                    "kpi-title"
                )

                total_value = ui.label(
                    "--"
                ).classes(
                    "kpi-value"
                )


            with ui.column().classes(
                "card-box flex-1"
            ):

                ui.label(
                    "AVERAGE DAILY SALES"
                ).classes(
                    "kpi-title"
                )

                average_value = ui.label(
                    "--"
                ).classes(
                    "kpi-value"
                )


            with ui.column().classes(
                "card-box flex-1"
            ):

                ui.label(
                    "PEAK DEMAND"
                ).classes(
                    "kpi-title"
                )

                peak_value = ui.label(
                    "--"
                ).classes(
                    "kpi-value"
                )


            with ui.column().classes(
                "card-box flex-1"
            ):

                ui.label(
                    "FORECAST TREND"
                ).classes(
                    "kpi-title"
                )

                trend_value = ui.label(
                    "--"
                ).classes(
                    "kpi-value"
                )


        # ====================================================
        # ACTUAL VS FORECAST
        # ====================================================

        with ui.column().classes(
            "card-box w-full mt-5"
        ):


            ui.label(
                "📈 Historical Actual vs Future Forecast"
            ).classes(
                "section-title"
            )


            actual_chart = ui.echart({

                "tooltip": {
                    "trigger":
                        "axis"
                },

                "legend": {

                    "data": [
                        "Historical Actual",
                        "Future Forecast"
                    ]

                },

                "xAxis": {

                    "type":
                        "category",

                    "data":
                        []

                },

                "yAxis": {

                    "type":
                        "value",

                    "name":
                        "Units Sold"

                },

                "series":
                    []

            }).classes(
                "w-full h-96"
            )


        # ====================================================
        # SCENARIO COMPARISON
        # ====================================================

        with ui.column().classes(
            "card-box w-full mt-5"
        ):


            ui.label(
                "🧪 Business Scenario Comparison"
            ).classes(
                "section-title"
            )


            ui.label(
                "Compare Base Case, Promotion, Higher Discount "
                "and Increased Marketing."
            ).classes(
                "text-gray-500"
            )


            scenario_chart = ui.echart({

                "tooltip": {
                    "trigger":
                        "axis"
                },

                "legend": {

                    "data": [
                        "Base Case",
                        "Promotion",
                        "Higher Discount",
                        "Higher Marketing"
                    ]

                },

                "xAxis": {

                    "type":
                        "category",

                    "data":
                        []

                },

                "yAxis": {

                    "type":
                        "value",

                    "name":
                        "Units Sold"

                },

                "series":
                    []

            }).classes(
                "w-full h-96"
            )


        # ====================================================
        # FORECAST INSIGHTS
        # ====================================================

        with ui.column().classes(
            "card-box w-full mt-5"
        ):


            ui.label(
                "🧠 Forecast Insights"
            ).classes(
                "section-title"
            )


            insight_1 = ui.label(
                "📊 Generate a forecast to see demand insights."
            ).classes(
                "insight-box"
            )


            insight_2 = ui.label(
                "📈 Trend analysis will appear here."
            ).classes(
                "insight-box"
            )


            insight_3 = ui.label(
                "🧪 Scenario recommendation will appear here."
            ).classes(
                "insight-box"
            )


            insight_4 = ui.label(
                "🔥 Peak-demand information will appear here."
            ).classes(
                "insight-box"
            )


        # ====================================================
        # REGIONAL PRODUCT RANKING
        # ====================================================

        with ui.column().classes(
            "card-box w-full mt-5"
        ):


            ui.label(
                "🏆 Regional Product Demand Ranking"
            ).classes(
                "section-title"
            )


            ranking_chart = ui.echart({

                "tooltip": {
                    "trigger":
                        "axis"
                },

                "xAxis": {

                    "type":
                        "value",

                    "name":
                        "Forecast Units"

                },

                "yAxis": {

                    "type":
                        "category",

                    "data":
                        []

                },

                "series": [

                    {

                        "type":
                            "bar",

                        "data":
                            []

                    }

                ]

            }).classes(
                "w-full h-96"
            )


            ranking_table = ui.table(

                columns=[

                    {
                        "name":
                            "rank",

                        "label":
                            "Rank",

                        "field":
                            "rank"
                    },

                    {
                        "name":
                            "product",

                        "label":
                            "Product",

                        "field":
                            "product"
                    },

                    {
                        "name":
                            "name",

                        "label":
                            "Product Name",

                        "field":
                            "name"
                    },

                    {
                        "name":
                            "units",

                        "label":
                            "Forecast Units",

                        "field":
                            "units"
                    }

                ],

                rows=[]

            ).classes(
                "w-full"
            )


# ============================================================
# REAL FORECAST CALLBACK
# ============================================================

def generate_real_forecast():

    try:

        product = product_map[
            product_select.value
        ]


        store = store_map[
            store_select.value
        ]


        start = pd.Timestamp(
            forecast_date.value
        )


        days = int(
            horizon.value
        )


        # ====================================================
        # BASE SETTINGS
        # ====================================================

        base_settings = {

            "Price":
                float(
                    price.value
                ),

            "Discount_Percentage":
                int(
                    discount.value
                ),

            "Promotion_Flag":
                1
                if promotion.value == "Yes"
                else 0,

            "Stock_Availability":
                1
                if stock.value == "Available"
                else 0,

            "Holiday_Flag":
                1
                if holiday.value == "Yes"
                else 0,

            "Local_Event_Flag":
                1
                if local_event.value == "Yes"
                else 0,

            "Competitor_Price":
                float(
                    competitor_price.value
                ),

            "Economic_Indicator":
                float(
                    economic_indicator.value
                ),

            "Marketing_Spend":
                float(
                    marketing.value
                )
        }


        base, history_df = forecast_product_store(

            product,
            store,
            start,
            days,
            base_settings

        )


        if base is None:

            ui.notify(
                "Not enough historical data for this Product / Store "
                "before the selected forecast date.",
                type="negative"
            )

            return


        # ====================================================
        # KPI
        # ====================================================

        total = (
            base["Predicted"]
            .sum()
        )


        average = (
            base["Predicted"]
            .mean()
        )


        peak = (
            base["Predicted"]
            .max()
        )


        first_week = (
            base["Predicted"]
            .head(7)
            .mean()
        )


        last_week = (
            base["Predicted"]
            .tail(7)
            .mean()
        )


        if first_week > 0:

            trend = (

                (
                    last_week
                    -
                    first_week
                )

                /
                first_week

                *
                100
            )

        else:

            trend = 0


        total_value.set_text(
            f"{total:,.0f} Units"
        )


        average_value.set_text(
            f"{average:.0f} Units"
        )


        peak_value.set_text(
            f"{peak:.0f} Units"
        )


        trend_value.set_text(
            f"{trend:+.1f}%"
        )


        # ====================================================
        # HISTORICAL + FORECAST CHART
        # ====================================================

        historical = (
            history_df
            .tail(40)
        )


        historical_dates = (

            historical["Date"]

            .dt.strftime(
                "%d-%m-%Y"
            )

            .tolist()
        )


        future_dates = (

            base["Date"]

            .dt.strftime(
                "%d-%m-%Y"
            )

            .tolist()
        )


        combined_dates = (
            historical_dates
            +
            future_dates
        )


        historical_values = (

            historical[
                "Units_Sold"
            ]

            .astype(float)

            .tolist()
        )


        actual_series = (

            historical_values

            +

            [None]
            *
            len(future_dates)
        )


        future_series = (

            [None]
            *
            len(historical_dates)

            +

            base[
                "Predicted"
            ]
            .round()
            .tolist()
        )


        actual_chart.options[
            "xAxis"
        ][
            "data"
        ] = combined_dates


        actual_chart.options[
            "series"
        ] = [

            {

                "name":
                    "Historical Actual",

                "type":
                    "line",

                "smooth":
                    True,

                "data":
                    actual_series

            },

            {

                "name":
                    "Future Forecast",

                "type":
                    "line",

                "smooth":
                    True,

                "data":
                    future_series,

                "areaStyle": {
                    "opacity": 0.08
                }

            }

        ]


        actual_chart.update()


        # ====================================================
        # PROMOTION SCENARIO
        # ====================================================

        promotion_settings = (
            base_settings.copy()
        )


        promotion_settings[
            "Promotion_Flag"
        ] = 1


        promotion_forecast, _ = forecast_product_store(

            product,
            store,
            start,
            days,
            promotion_settings

        )


        # ====================================================
        # HIGHER DISCOUNT SCENARIO
        # ====================================================

        discount_settings = (
            base_settings.copy()
        )


        discount_settings[
            "Discount_Percentage"
        ] = min(

            100,

            int(
                discount.value
            )
            +
            10
        )


        discount_forecast, _ = forecast_product_store(

            product,
            store,
            start,
            days,
            discount_settings

        )


        # ====================================================
        # HIGHER MARKETING SCENARIO
        # ====================================================

        marketing_settings = (
            base_settings.copy()
        )


        marketing_settings[
            "Marketing_Spend"
        ] = (

            float(
                marketing.value
            )

            *
            1.50
        )


        marketing_forecast, _ = forecast_product_store(

            product,
            store,
            start,
            days,
            marketing_settings

        )


        # ====================================================
        # SCENARIO CHART
        # ====================================================

        scenario_chart.options[
            "xAxis"
        ][
            "data"
        ] = future_dates


        scenario_chart.options[
            "series"
        ] = [

            {

                "name":
                    "Base Case",

                "type":
                    "line",

                "smooth":
                    True,

                "data":
                    base[
                        "Predicted"
                    ]
                    .round()
                    .tolist()

            },

            {

                "name":
                    "Promotion",

                "type":
                    "line",

                "smooth":
                    True,

                "data":
                    promotion_forecast[
                        "Predicted"
                    ]
                    .round()
                    .tolist()

            },

            {

                "name":
                    "Higher Discount",

                "type":
                    "line",

                "smooth":
                    True,

                "data":
                    discount_forecast[
                        "Predicted"
                    ]
                    .round()
                    .tolist()

            },

            {

                "name":
                    "Higher Marketing",

                "type":
                    "line",

                "smooth":
                    True,

                "data":
                    marketing_forecast[
                        "Predicted"
                    ]
                    .round()
                    .tolist()

            }

        ]


        scenario_chart.update()


        # ====================================================
        # BUSINESS INSIGHTS
        # ====================================================

        product_name = (
            history_df[
                "Product_Name"
            ]
            .iloc[-1]
        )


        insight_1.set_text(

            f"📊 {product_name} is forecast to sell approximately "
            f"{total:,.0f} units during the next {days} days."
        )


        if trend > 5:

            trend_text = (
                "upward"
            )

        elif trend < -5:

            trend_text = (
                "downward"
            )

        else:

            trend_text = (
                "stable"
            )


        insight_2.set_text(

            f"📈 Demand is showing a {trend_text} trend. "
            f"The difference between the first and final forecast "
            f"week is {trend:+.1f}%."
        )


        scenario_totals = {

            "Promotion":
                promotion_forecast[
                    "Predicted"
                ].sum(),

            "Higher Discount":
                discount_forecast[
                    "Predicted"
                ].sum(),

            "Higher Marketing":
                marketing_forecast[
                    "Predicted"
                ].sum()

        }


        best_scenario = max(
            scenario_totals,
            key=scenario_totals.get
        )


        best_total = scenario_totals[
            best_scenario
        ]


        if total > 0:

            uplift = (

                (
                    best_total
                    -
                    total
                )

                /
                total

                *
                100
            )

        else:

            uplift = 0


        insight_3.set_text(

            f"🧪 Best tested scenario: {best_scenario}. "
            f"Forecast demand becomes {best_total:,.0f} units "
            f"({uplift:+.1f}% compared with Base Case)."
        )


        peak_row = (

            base.loc[
                base[
                    "Predicted"
                ].idxmax()
            ]

        )


        insight_4.set_text(

            f"🔥 Peak predicted demand is "
            f"{peak_row['Predicted']:.0f} units on "
            f"{peak_row['Date'].strftime('%d-%m-%Y')}."
        )


        status.set_text(

            f"✅ Real XGBoost forecast generated for "
            f"{product_name} at {store}."
        )


        ui.notify(
            "Forecast generated successfully!",
            type="positive"
        )


    except Exception as error:

        print(
            "\nFORECAST ERROR:"
        )

        traceback.print_exc()


        ui.notify(
            f"Forecast error: {error}",
            type="negative"
        )


# ============================================================
# REGIONAL PRODUCT RANKING
# ============================================================

def generate_product_ranking():

    try:

        region = region_select.value


        start = pd.Timestamp(
            forecast_date.value
        )


        days = int(
            horizon.value
        )


        region_df = df[
            df[
                "Store_Location"
            ] == region
        ]


        combinations = (

            region_df[
                [
                    "Product_ID",
                    "Store_ID",
                    "Product_Name"
                ]
            ]

            .drop_duplicates()
        )


        totals = {}


        for _, row in combinations.iterrows():


            product = row[
                "Product_ID"
            ]


            store = row[
                "Store_ID"
            ]


            forecast_result, _ = forecast_product_store(

                product,
                store,
                start,
                days

            )


            if forecast_result is None:

                continue


            forecast_units = (

                forecast_result[
                    "Predicted"
                ]

                .sum()
            )


            if product not in totals:

                totals[
                    product
                ] = {

                    "name":
                        row[
                            "Product_Name"
                        ],

                    "units":
                        0
                }


            totals[
                product
            ][
                "units"
            ] += forecast_units


        ranking = [

            {

                "product":
                    product,

                "name":
                    values[
                        "name"
                    ],

                "units":
                    round(
                        values[
                            "units"
                        ]
                    )

            }

            for product, values
            in totals.items()

        ]


        ranking = sorted(

            ranking,

            key=lambda item:
                item[
                    "units"
                ],

            reverse=True
        )


        for index, item in enumerate(
            ranking,
            start=1
        ):

            item[
                "rank"
            ] = index


        ranking_table.rows = ranking

        ranking_table.update()


        top = ranking[:8]


        ranking_chart.options[
            "yAxis"
        ][
            "data"
        ] = [

            item[
                "product"
            ]

            for item in reversed(top)

        ]


        ranking_chart.options[
            "series"
        ][0][
            "data"
        ] = [

            item[
                "units"
            ]

            for item in reversed(top)

        ]


        ranking_chart.update()


        ui.notify(
            f"Product ranking generated for {region}",
            type="positive"
        )


    except Exception as error:

        print(
            "\nRANKING ERROR:"
        )

        traceback.print_exc()


        ui.notify(
            f"Ranking error: {error}",
            type="negative"
        )


# ============================================================
# BUTTON CONNECTIONS
# ============================================================

generate_button.on(
    "click",
    generate_real_forecast
)


ranking_button.on(
    "click",
    generate_product_ranking
)


# ============================================================
# START APP
# ============================================================

import os

PORT = int(os.environ.get("PORT", 8092))

ui.run(
    title="Sales Forecast Intelligence",
    host="0.0.0.0",
    port=PORT,
    reload=False,
    show=False
)
