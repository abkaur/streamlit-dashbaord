import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# --------------------------------------------------
# Part 2 - Load Dataset
# --------------------------------------------------

st.title("EC2 Instance EDA Dashboard")

st.markdown(
    """
    **Cloud Economics — Lab 3**  
    **Student:** Abhijot Kaur  
    **EC2 Instance Cost & Specification Explorer**
    """
)

st.divider()

df = pd.read_csv("ec2dataset.csv")

st.write("Dataset Preview")
st.dataframe(df)


# --------------------------------------------------
# Part 3 - Dataset Information
# --------------------------------------------------

st.subheader("Dataset Information")

col1, col2, col3 = st.columns(3)

col1.metric("Number of Instances", len(df))
col2.metric("Number of Columns", len(df.columns))
col3.metric("Missing Values", df.isna().sum().sum())


# --------------------------------------------------
# Part 4 - Dataset Structure
# --------------------------------------------------

st.subheader("Dataset Structure")

st.write("Columns:")
st.write(df.columns.tolist())

st.subheader("Data Types")
st.write(df.dtypes)


# --------------------------------------------------
# Part 5 - Clean Memory
# --------------------------------------------------

df["Memory_GiB"] = (
    df["Instance Memory"]
    .str.extract(r"([\d.]+)")
    .astype(float)
)


# --------------------------------------------------
# Part 6 - Clean vCPU
# --------------------------------------------------

df["vCPU_Count"] = (
    df["vCPUs"]
    .str.extract(r"(\d+)")
    .astype(float)
)


# --------------------------------------------------
# Part 7 - Clean Pricing Data
# --------------------------------------------------

def clean_price(value):

    if pd.isna(value):
        return None

    value = str(value)

    if "unavailable" in value.lower():
        return None

    return float(
        value.replace("$", "")
        .replace(" hourly", "")
        .strip()
    )


price_columns = [
    "On Demand",
    "Linux Reserved cost",
    "Linux Spot Minimum cost",
    "Windows On Demand cost",
    "Windows Reserved cost"
]


for column in price_columns:
    df[column + "_USD"] = df[column].apply(clean_price)


# --------------------------------------------------
# Part 8 - Monthly On-Demand Cost
# --------------------------------------------------

df["Monthly_On_Demand"] = df["On Demand_USD"] * 730

st.subheader("Monthly On-Demand Cost")

st.dataframe(
    df[
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
)


# --------------------------------------------------
# Part 9 + Part 10 - Interactive Filters
# --------------------------------------------------

st.sidebar.header("Filters")

memory_values = sorted(
    df["Memory_GiB"].dropna().unique()
)

memory_filter = st.sidebar.select_slider(
    "Maximum Memory (GiB)",
    options=memory_values,
    value=memory_values[-1]
)

cpu_values = sorted(
    df["vCPU_Count"]
    .dropna()
    .unique()
)

selected_cpu = st.sidebar.multiselect(
    "vCPU Count",
    options=cpu_values,
    default=cpu_values
)
# Challenge 1 - Network Performance Filter

network_values = sorted(
    df["Network Performance"].dropna().unique()
)

selected_network = st.sidebar.multiselect(
    "Network Performance",
    options=network_values,
    default=network_values
)

storage_values = sorted(
    df["Instance Storage"].dropna().unique()
)

selected_storage = st.sidebar.multiselect(
    "Instance Storage",
    options=storage_values,
    default=storage_values
)


# Challenge - Maximum Hourly Price

hourly_price_values = sorted(
    df["On Demand_USD"].dropna().unique()
)

hourly_price_filter = st.sidebar.select_slider(
    "Maximum Hourly Price ($)",
    options=hourly_price_values,
    value=hourly_price_values[-1]
)
# Challenge - Maximum Monthly Cost

monthly_cost_values = sorted(
    df["Monthly_On_Demand"].dropna().unique()
)

monthly_cost_filter = st.sidebar.select_slider(
    "Maximum Monthly Cost ($)",
    options=monthly_cost_values,
    value=monthly_cost_values[-1]
)

filtered_df = df[
    (df["Memory_GiB"] <= memory_filter)
    &
    (df["vCPU_Count"].isin(selected_cpu))
    &
    (df["Network Performance"].isin(selected_network))
    &
    (df["Instance Storage"].isin(selected_storage))
    &
    (df["On Demand_USD"] <= hourly_price_filter)
    &
    (df["Monthly_On_Demand"] <= monthly_cost_filter)
]


st.subheader("Filtered EC2 Instances")

st.write(
    f"{len(filtered_df)} instances found"
)

st.dataframe(filtered_df)


# --------------------------------------------------
# Part 11 - KPI Cards
# --------------------------------------------------

st.subheader("EC2 Summary")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Instances",
    len(filtered_df)
)

col2.metric(
    "Avg Memory",
    f"{filtered_df['Memory_GiB'].mean():.2f} GiB"
)

col3.metric(
    "Avg vCPUs",
    f"{filtered_df['vCPU_Count'].mean():.1f}"
)

col4.metric(
    "Avg Hourly Cost",
    f"${filtered_df['On Demand_USD'].mean():.4f}"
)


# --------------------------------------------------
# Part 12 - Memory Distribution
# --------------------------------------------------

st.subheader("Memory Distribution")

fig = px.histogram(
    filtered_df,
    x="Memory_GiB",
    nbins=30,
    title="Distribution of EC2 Memory"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="memory_distribution"
)


if not filtered_df.empty:

    most_common_memory = filtered_df["Memory_GiB"].mode()[0]

    median_memory = filtered_df["Memory_GiB"].median()

    max_memory_value = filtered_df["Memory_GiB"].max()

    st.write("**Insights:**")

    st.write(
        f"- The most common memory size is approximately "
        f"**{most_common_memory:.1f} GiB**."
    )

    st.write(
        f"- The median memory is "
        f"**{median_memory:.1f} GiB**."
    )

    st.write(
        f"- The largest instance in the current filtered data has "
        f"**{max_memory_value:.1f} GiB** of memory."
    )


# --------------------------------------------------
# Part 13 - vCPU Distribution
# --------------------------------------------------

st.subheader("vCPU Distribution")

fig = px.histogram(
    filtered_df,
    x="vCPU_Count",
    title="Distribution of vCPUs"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="vcpu_distribution"
)


if not filtered_df.empty:

    most_common_cpu = filtered_df["vCPU_Count"].mode()[0]

    small_cpu_count = filtered_df[
        filtered_df["vCPU_Count"].between(1, 4)
    ].shape[0]

    st.write("**Insights:**")

    st.write(
        f"- The most common CPU configuration is "
        f"**{most_common_cpu:.0f} vCPUs**."
    )

    st.write(
        f"- **{small_cpu_count} instances** have between "
        f"1 and 4 vCPUs."
    )

    st.write(
        "- Changing the memory or vCPU filters updates "
        "this distribution automatically."
    )


# --------------------------------------------------
# Part 14 - Memory vs vCPU
# --------------------------------------------------

st.subheader("Memory vs vCPUs")

fig = px.scatter(
    filtered_df,
    x="vCPU_Count",
    y="Memory_GiB",
    hover_name="API Name",
    hover_data=["On Demand_USD"],
    title="EC2 Memory vs vCPU"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="memory_vs_cpu"
)


if len(filtered_df) > 1:

    correlation_cpu_memory = filtered_df[
        ["vCPU_Count", "Memory_GiB"]
    ].corr().iloc[0, 1]

    st.write("**Insight:**")

    if correlation_cpu_memory > 0.5:

        st.write(
            f"- There is a clear positive relationship between "
            f"vCPUs and memory "
            f"(correlation ≈ **{correlation_cpu_memory:.2f}**). "
            f"Instances with more vCPUs generally have more memory."
        )

    elif correlation_cpu_memory > 0:

        st.write(
            f"- There is a positive relationship between vCPUs "
            f"and memory "
            f"(correlation ≈ **{correlation_cpu_memory:.2f}**), "
            f"although it is not very strong."
        )

    else:

        st.write(
            "- The filtered data does not show a clear positive "
            "relationship between vCPUs and memory."
        )


# --------------------------------------------------
# Part 15 - Memory vs On-Demand Cost
# --------------------------------------------------

st.subheader("Memory vs On-Demand Cost")

fig = px.scatter(
    filtered_df,
    x="Memory_GiB",
    y="On Demand_USD",
    hover_name="API Name",
    size="vCPU_Count",
    title="Memory vs EC2 On-Demand Cost"
)

st.plotly_chart(
    fig,
    use_container_width=True,
    key="memory_vs_cost"
)


if len(filtered_df) > 1:

    correlation_memory_cost = filtered_df[
        ["Memory_GiB", "On Demand_USD"]
    ].corr().iloc[0, 1]

    st.write("**Insight:**")

    if correlation_memory_cost > 0.5:

        st.write(
            f"- Memory and on-demand cost show a positive relationship "
            f"(correlation ≈ **{correlation_memory_cost:.2f}**). "
            f"Instances with more memory generally cost more."
        )

    elif correlation_memory_cost > 0:

        st.write(
            f"- Memory and cost have a positive relationship "
            f"(correlation ≈ **{correlation_memory_cost:.2f}**), "
            f"but memory is not the only factor affecting EC2 price."
        )

    else:

        st.write(
            "- The filtered data does not show a clear positive "
            "relationship between memory and on-demand cost."
        )

# --------------------------------------------------
# Part 16 - Lowest-Cost EC2 Instances
# --------------------------------------------------

st.subheader("Lowest-Cost EC2 Instances")

cheapest = (
    filtered_df
    .sort_values("On Demand_USD")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(cheapest)

if not cheapest.empty:
    cheapest_instance = cheapest.iloc[0]

    st.write(
        f"**Insight:** The lowest-cost instance in the current filtered data is "
        f"**{cheapest_instance['API Name']}** at approximately "
        f"**${cheapest_instance['On Demand_USD']:.4f}/hour**."
    )


# --------------------------------------------------
# Part 17 - Highest-Cost EC2 Instances
# --------------------------------------------------

st.subheader("Highest-Cost EC2 Instances")

most_expensive = (
    filtered_df
    .sort_values(
        "On Demand_USD",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Monthly_On_Demand"
        ]
    ]
    .head(10)
)

st.dataframe(most_expensive)

if not most_expensive.empty:
    expensive_instance = most_expensive.iloc[0]

    st.write(
        f"**Insight:** The highest-cost instance in the current filtered data is "
        f"**{expensive_instance['API Name']}** at approximately "
        f"**${expensive_instance['On Demand_USD']:.4f}/hour**."
    )


# --------------------------------------------------
# Part 18 - EC2 Pricing Comparison
# --------------------------------------------------

st.subheader("EC2 Pricing Comparison")

pricing = filtered_df[
    [
        "Name",
        "API Name",
        "On Demand_USD",
        "Linux Reserved cost_USD",
        "Linux Spot Minimum cost_USD",
        "Windows On Demand cost_USD",
        "Windows Reserved cost_USD"
    ]
]

st.dataframe(pricing)


# --------------------------------------------------
# Part 19 - Pricing Model Chart
# --------------------------------------------------

st.subheader("Pricing Model Comparison")

if not filtered_df.empty:

    selected_instance = st.selectbox(
        "Select an EC2 Instance",
        filtered_df["API Name"].unique(),
        key="pricing_instance_selection"
    )

    instance = filtered_df[
        filtered_df["API Name"] == selected_instance
    ].iloc[0]

    pricing_data = pd.DataFrame(
        {
            "Pricing Model": [
                "On Demand",
                "Linux Reserved",
                "Linux Spot"
            ],

            "Hourly Cost": [
                instance["On Demand_USD"],
                instance["Linux Reserved cost_USD"],
                instance["Linux Spot Minimum cost_USD"]
            ]
        }
    )

    pricing_data = pricing_data.dropna()

    fig = px.bar(
        pricing_data,
        x="Pricing Model",
        y="Hourly Cost",
        title=f"Pricing Comparison: {selected_instance}"
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        key="pricing_comparison_chart"
    )


    # --------------------------------------------------
    # Part 20 - Monthly Pricing
    # --------------------------------------------------

    pricing_data["Monthly Cost"] = (
        pricing_data["Hourly Cost"] * 730
    )

    st.subheader("Monthly Pricing Comparison")

    st.dataframe(pricing_data)

    if not pricing_data.empty:

        cheapest_model = pricing_data.loc[
            pricing_data["Hourly Cost"].idxmin()
        ]

        st.write(
            f"**Insight:** For **{selected_instance}**, "
            f"**{cheapest_model['Pricing Model']}** is the lowest-cost "
            f"pricing option among the available models shown here."
        )


# --------------------------------------------------
# Part 21 - Cost Efficiency Metric
# --------------------------------------------------

st.subheader("Cost per GiB of Memory")

filtered_df = filtered_df.copy()

filtered_df["Cost_Per_GiB"] = (
    filtered_df["On Demand_USD"]
    /
    filtered_df["Memory_GiB"]
)

efficiency = (
    filtered_df
    .sort_values("Cost_Per_GiB")
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "On Demand_USD",
            "Cost_Per_GiB"
        ]
    ]
    .head(15)
)

st.dataframe(efficiency)

if not efficiency.empty:

    efficient_instance = efficiency.iloc[0]

    st.write(
        f"**Insight:** Based only on cost per GiB, "
        f"**{efficient_instance['API Name']}** has one of the lowest "
        f"cost-per-memory values in the current filtered results."
    )

    st.caption(
        "A lower cost per GiB does not automatically mean an instance "
        "is the best choice. CPU, storage, networking and workload "
        "requirements also matter."
    )

# --------------------------------------------------
# Challenge 5 - Memory per vCPU
# --------------------------------------------------

filtered_df = filtered_df.copy()

filtered_df["Memory_per_vCPU"] = (
    filtered_df["Memory_GiB"] /
    filtered_df["vCPU_Count"]
)

st.subheader("Memory per vCPU")

memory_cpu_ratio = (
    filtered_df
    .sort_values(
        "Memory_per_vCPU",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "Memory_per_vCPU"
        ]
    ]
    .head(15)
)

st.dataframe(memory_cpu_ratio)

if not memory_cpu_ratio.empty:

    highest_memory_cpu = memory_cpu_ratio.iloc[0]

    st.write(
        f"**Answer:** In the current filtered results, "
        f"**{highest_memory_cpu['API Name']}** has the highest "
        f"memory-per-vCPU ratio at approximately "
        f"**{highest_memory_cpu['Memory_per_vCPU']:.2f} GiB per vCPU**."
    )

    st.write(
        "A higher memory-per-vCPU ratio may be useful for "
        "memory-intensive workloads, but it does not automatically "
        "make the instance the best or lowest-cost choice."
    )

st.subheader("Memory per vCPU")

memory_cpu_ratio = (
    filtered_df
    .sort_values(
        "Memory_per_vCPU",
        ascending=False
    )
    [
        [
            "Name",
            "API Name",
            "Memory_GiB",
            "vCPU_Count",
            "Memory_per_vCPU"
        ]
    ]
    .head(15)
)

st.dataframe(memory_cpu_ratio)

if not memory_cpu_ratio.empty:
    top_ratio_instance = memory_cpu_ratio.iloc[0]

    st.write(
        f"**Insight:** {top_ratio_instance['API Name']} has one of the "
        f"highest memory-per-vCPU ratios in the current filtered results."
    )

# --------------------------------------------------
# Challenge Insights
# --------------------------------------------------

st.header("Challenge Analysis")

st.write(
    "The additional filters allow the EC2 dataset to be explored "
    "based on technical requirements and cost constraints."
)

st.subheader("Network Performance")

if selected_network:
    st.write(
        f"**Insight:** The dashboard is currently filtering EC2 instances "
        f"using {len(selected_network)} selected network-performance "
        f"option(s). Changing this filter helps identify instances that "
        f"meet specific networking requirements."
    )

st.subheader("Instance Storage")

if selected_storage:
    st.write(
        f"**Insight:** The storage filter allows EC2 instances to be "
        f"compared based on their local storage configuration. "
        f"The current selection includes {len(selected_storage)} "
        f"storage option(s)."
    )

st.subheader("Maximum Hourly Price")

st.write(
    f"**Insight:** The current hourly budget is "
    f"**${hourly_price_filter:.4f} per hour**. "
    f"Instances above this price are excluded from the results."
)

st.subheader("Maximum Monthly Cost")

st.write(
    f"**Insight:** The current monthly budget is "
    f"**${monthly_cost_filter:,.2f}** based on 730 hours per month. "
    f"Instances above this estimated monthly cost are removed."
)

st.header("Overall Analysis")

st.write(
    """
    This dashboard shows that EC2 instance selection should not be
    based on price alone. Memory, vCPU count, network performance,
    storage requirements, and pricing model all affect the final
    choice. The interactive filters make it possible to narrow the
    dataset according to both technical requirements and budget.
    """
)
# ==================================================
# NEW LAB - EC2 COST ANALYSIS AND REGRESSION
# ==================================================

st.divider()
st.header("EC2 Cost Analysis and Regression")

st.write(
    """
    This section analyzes EC2 pricing trends, identifies cost outliers,
    compares instance families, and uses linear regression to predict
    On-Demand EC2 pricing.
    """
)


# --------------------------------------------------
# Cost Summary
# --------------------------------------------------

st.subheader("Cost Summary Statistics")

analysis_cost_columns = [
    "On Demand_USD",
    "Linux Reserved cost_USD",
    "Linux Spot Minimum cost_USD",
    "Windows On Demand cost_USD",
    "Windows Reserved cost_USD"
]

st.dataframe(
    df[analysis_cost_columns].describe()
)


# --------------------------------------------------
# Check Invalid Negative Prices
# --------------------------------------------------

st.subheader("Negative Price Check")

negative_results = {}

for column in analysis_cost_columns:
    negative_results[column] = int(
        (df[column] < 0).sum()
    )

negative_df = pd.DataFrame(
    list(negative_results.items()),
    columns=["Pricing Column", "Negative Values"]
)

st.dataframe(negative_df)

if negative_df["Negative Values"].sum() == 0:
    st.success("No negative pricing values were found in the dataset.")


# --------------------------------------------------
# Outlier Analysis
# --------------------------------------------------

st.subheader("On-Demand Cost Outlier Analysis")

outlier_data = df.dropna(
    subset=["On Demand_USD"]
).copy()

Q1 = outlier_data["On Demand_USD"].quantile(0.25)
Q3 = outlier_data["On Demand_USD"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

on_demand_outliers = outlier_data[
    (outlier_data["On Demand_USD"] < lower_bound)
    |
    (outlier_data["On Demand_USD"] > upper_bound)
]

cleaned_analysis = outlier_data[
    (outlier_data["On Demand_USD"] >= lower_bound)
    &
    (outlier_data["On Demand_USD"] <= upper_bound)
].copy()


col1, col2, col3 = st.columns(3)

col1.metric(
    "Valid On-Demand Rows",
    len(outlier_data)
)

col2.metric(
    "Outliers Detected",
    len(on_demand_outliers)
)

col3.metric(
    "Rows After Removal",
    len(cleaned_analysis)
)

st.write(
    f"IQR lower bound: **${lower_bound:.4f}**"
)

st.write(
    f"IQR upper bound: **${upper_bound:.4f}**"
)


# --------------------------------------------------
# Before Outlier Removal
# --------------------------------------------------

st.subheader("Cost Distribution Before Outlier Removal")

fig_before, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=df[analysis_cost_columns],
    ax=ax
)

ax.set_title(
    "Cost Comparison of Amazon EC2 Instances (Hourly)"
)

ax.set_ylabel("Cost (USD)")
ax.tick_params(axis="x", rotation=45)

st.pyplot(fig_before)

plt.close(fig_before)


# --------------------------------------------------
# After Outlier Removal
# --------------------------------------------------

st.subheader("Cost Distribution After On-Demand Outlier Removal")

fig_after, ax = plt.subplots(figsize=(12, 6))

sns.boxplot(
    data=cleaned_analysis[analysis_cost_columns],
    ax=ax
)

ax.set_title(
    "EC2 Cost Distribution After Removing On-Demand Outliers"
)

ax.set_ylabel("Cost (USD)")
ax.tick_params(axis="x", rotation=45)

st.pyplot(fig_after)

plt.close(fig_after)


# --------------------------------------------------
# Instance Family Analysis
# --------------------------------------------------

st.subheader("T2 vs T3 Instance Family Analysis")

t2_instances = cleaned_analysis[
    cleaned_analysis["Name"]
    .str.upper()
    .str.startswith("T2", na=False)
]

t3_instances = cleaned_analysis[
    cleaned_analysis["Name"]
    .str.upper()
    .str.startswith("T3", na=False)
]


family_col1, family_col2 = st.columns(2)

family_col1.metric(
    "T2 Instances",
    len(t2_instances)
)

family_col2.metric(
    "T3 Instances",
    len(t3_instances)
)


st.write("### T2 Cost Summary")

st.dataframe(
    t2_instances[
        analysis_cost_columns
    ].describe()
)


st.write("### T3 Cost Summary")

st.dataframe(
    t3_instances[
        analysis_cost_columns
    ].describe()
)


# --------------------------------------------------
# Lowest Cost T2/T3 Instances
# --------------------------------------------------

st.subheader("Lowest-Cost T2/T3 Instances")

family_comparison = pd.concat(
    [
        t2_instances[
            [
                "Name",
                "On Demand_USD",
                "Linux Reserved cost_USD"
            ]
        ],
        t3_instances[
            [
                "Name",
                "On Demand_USD",
                "Linux Reserved cost_USD"
            ]
        ]
    ]
)

family_comparison = (
    family_comparison
    .dropna()
    .sort_values("On Demand_USD")
    .head(10)
)

st.dataframe(family_comparison)


# --------------------------------------------------
# Regression Analysis
# --------------------------------------------------

st.subheader("EC2 On-Demand Cost Prediction")

regression_data = cleaned_analysis[
    [
        "Memory_GiB",
        "vCPU_Count",
        "On Demand_USD"
    ]
].dropna().copy()


X = regression_data[
    [
        "Memory_GiB",
        "vCPU_Count"
    ]
]

y = regression_data[
    "On Demand_USD"
]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


model = LinearRegression()

model.fit(
    X_train,
    y_train
)


y_pred = model.predict(
    X_test
)


mae = mean_absolute_error(
    y_test,
    y_pred
)

mse = mean_squared_error(
    y_test,
    y_pred
)

rmse = mse ** 0.5

r2 = r2_score(
    y_test,
    y_pred
)


st.write("### Model Performance")

metric1, metric2, metric3, metric4 = st.columns(4)

metric1.metric(
    "MAE",
    f"{mae:.4f}"
)

metric2.metric(
    "MSE",
    f"{mse:.4f}"
)

metric3.metric(
    "RMSE",
    f"{rmse:.4f}"
)

metric4.metric(
    "R² Score",
    f"{r2:.4f}"
)


st.write(
    f"""
    The regression model uses **Memory (GiB)** and **vCPU count**
    to predict the hourly On-Demand EC2 cost.

    The model achieved an **R² score of {r2:.2f}**.
    """
)


# --------------------------------------------------
# Actual vs Predicted
# --------------------------------------------------

st.subheader("Actual vs Predicted On-Demand Cost")

prediction_results = pd.DataFrame(
    {
        "Actual Cost": y_test,
        "Predicted Cost": y_pred
    }
)

fig_prediction = px.scatter(
    prediction_results,
    x="Actual Cost",
    y="Predicted Cost",
    title="Actual vs Predicted EC2 On-Demand Cost"
)

fig_prediction.add_shape(
    type="line",
    x0=prediction_results["Actual Cost"].min(),
    y0=prediction_results["Actual Cost"].min(),
    x1=prediction_results["Actual Cost"].max(),
    y1=prediction_results["Actual Cost"].max(),
    line=dict(dash="dash")
)

st.plotly_chart(
    fig_prediction,
    use_container_width=True
)


# --------------------------------------------------
# Interactive Cost Prediction
# --------------------------------------------------

st.subheader("Predict EC2 On-Demand Cost")

prediction_col1, prediction_col2 = st.columns(2)

with prediction_col1:

    prediction_memory = st.number_input(
        "Memory (GiB)",
        min_value=0.5,
        value=4.0,
        step=0.5
    )


with prediction_col2:

    prediction_vcpu = st.number_input(
        "vCPUs",
        min_value=1,
        value=2,
        step=1
    )


new_instance = pd.DataFrame(
    [
        {
            "Memory_GiB": prediction_memory,
            "vCPU_Count": prediction_vcpu
        }
    ]
)


predicted_hourly_cost = model.predict(
    new_instance
)[0]


st.metric(
    "Predicted On-Demand Hourly Cost",
    f"${predicted_hourly_cost:.4f}"
)


st.write(
    f"""
    Estimated monthly cost at 730 hours:

    **${predicted_hourly_cost * 730:,.2f}**
    """
)
# --------------------------------------------------
# Part 22 - Download Filtered Dataset
# --------------------------------------------------

st.subheader("Download Filtered Dataset")

csv = filtered_df.to_csv(index=False)

st.download_button(
    label="Download Filtered Dataset",
    data=csv,
    file_name="filtered_ec2_instances.csv",
    mime="text/csv"
)


# --------------------------------------------------
# Part 23 - Dashboard Complete
# --------------------------------------------------

st.divider()

st.success(
    "EC2 EDA Dashboard sections complete. "
    "Use the sidebar filters to explore EC2 specifications and costs."
)
