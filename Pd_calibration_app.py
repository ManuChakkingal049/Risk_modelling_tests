# pd_calibration_app.py
"""
PD Calibration Accuracy Testing App (Streamlit)

Features:
- Conceptual explanation of calibration vs discrimination
- Chi-square goodness-of-fit test for multi-grade calibration
- Exact binomial test for single-grade calibration (exact)
- Jeffreys interval (Bayesian) for interval-based checks
- Worked examples and 'monitoring over time' demo
- Interactive inputs + visualizations
"""

import streamlit as st
import numpy as np
import pandas as pd
from math import isclose
from scipy import stats
from scipy.stats import beta
import matplotlib.pyplot as plt

st.set_page_config(page_title="PD Calibration Tests", layout="wide")

# -------------------------
# Helper functions
# -------------------------
def binomial_test_exact(k, n, p0, alternative='two-sided'):
    """
    Perform exact binomial test.
    Uses scipy.stats.binom_test if available (deprecated in newer versions),
    otherwise compute tails using binomial cdf/manually.
    alternative: 'two-sided', 'greater', 'less'
    Returns p-value.
    """
    try:
        # SciPy deprecated binom_test but many environments still have it
        pval = stats.binom_test(k, n=n, p=p0, alternative=alternative)
        return pval
    except AttributeError:
        # fallback implementation using cdf
        # Note: two-sided p-value is tricky for discrete distribution; use doubling min-tail approach
        pmf = stats.binom.pmf(k, n, p0)
        if alternative == 'greater':
            pval = stats.binom.sf(k-1, n, p0)  # P(X >= k)
        elif alternative == 'less':
            pval = stats.binom.cdf(k, n, p0)   # P(X <= k)
        else:
            # two-sided: sum probabilities of outcomes with <= pmf(k)
            probs = stats.binom.pmf(range(0, n+1), n, p0)
            pval = probs[probs <= pmf].sum()
            pval = min(1.0, pval)
        return pval

def jeffreys_interval(k, n, alpha=0.05):
    """
    Jeffreys interval (approx Bayesian credible interval with Jeffreys prior Beta(0.5,0.5)).
    Posterior = Beta(k+0.5, n-k+0.5)
    Returns (lower, upper)
    """
    a = k + 0.5
    b = n - k + 0.5
    lower = beta.ppf(alpha/2, a, b)
    upper = beta.ppf(1 - alpha/2, a, b)
    return lower, upper

def chi_square_calibration_test(obs_defaults, exposures, expected_pds):
    """
    Chi-square goodness-of-fit test for calibration across bins/grades.
    obs_defaults: list of observed default counts per grade (k_i)
    exposures: list of exposures per grade (n_i)
    expected_pds: list of model PDs per grade (p_i expected)
    expected_defaults = n_i * p_i
    Returns: chi2_stat, p_value, expected_defaults (array)
    """
    obs = np.array(obs_defaults, dtype=float)
    exp = np.array(exposures, dtype=float) * np.array(expected_pds, dtype=float)
    # If any expected are zero, chi-square undefined -> handle separately
    mask_zero = exp == 0
    if mask_zero.any():
        # Remove zero-expected bins for test but warn
        chi2_stat, p_value = np.nan, np.nan
        return chi2_stat, p_value, exp
    # Use scipy.stats.chisquare - degrees of freedom = k - 1 (no parameters estimated)
    chi2_stat, p_value = stats.chisquare(f_obs=obs, f_exp=exp)
    return chi2_stat, p_value, exp

def interpret_pvalue(p, alpha=0.05):
    if np.isnan(p):
        return "Test not applicable (invalid expected counts)"
    if p < alpha:
        return "Reject H₀ → evidence of mis-calibration (statistically significant)."
    else:
        return "Fail to reject H₀ → no statistical evidence of mis-calibration."

def traffic_light_from_p(p, alpha=0.05):
    if np.isnan(p):
        return "NA"
    if p < alpha:
        return "🔴 Mis-calibrated"
    elif p < 0.1:
        return "🟡 Borderline"
    else:
        return "🟢 Well-calibrated"

# -------------------------
# UI: header and layout
# -------------------------
st.title("PD Calibration Accuracy — Interactive App")
st.markdown("""
This app helps you **test PD calibration** (expected vs observed defaults) using:
- **Exact binomial test** (single-grade, exact),
- **Jeffreys interval** (Bayesian interval, helpful for small samples),
- **Chi-square goodness-of-fit** (multi-grade).  

It also explains **when to use each test**, shows worked examples, and demonstrates **monitoring over time**.
""")

tabs = st.tabs(["Concepts", "Tests & When to Use", "Interactive Tests", "Worked Examples", "Monitoring / Time Series"])

# -------------------------
# Tab 1: Concepts
# -------------------------
with tabs[0]:
    st.header("What is Calibration (and why it matters)?")
    st.markdown("""
    **Calibration** of a PD model means that the *predicted probabilities of default* correspond to the *observed frequencies* of default.
    - If a model predicts PD = 2% for a grade, then out of 100 obligors in that grade, ~2 should default *on average*.
    - **Calibration** is different from **discrimination** (ability to rank/order risk).
    
    **Why good calibration matters**
    - **Capital & provisioning:** PDs feed into expected loss (EL) and regulatory capital; miscalibrated PDs bias capital estimates.
    - **Pricing:** Underpricing or overpricing of loans if true risk differs.
    - **Business decisions:** Risk appetite, limit setting, and portfolio management rely on accurate PDs.
    - **Regulatory compliance:** Regulators require model validation and stability monitoring.

    **Real-life analogy**
    - A weather forecast saying "30% chance of rain" is well-calibrated if on days with that forecast it rains ~30% of the time. If it rains 70% of those days, the forecast is poorly calibrated and you get wet unexpectedly.
    """)

    st.subheader("When to test calibration (monitoring)")
    st.markdown("""
    - At model validation time (initial backtesting).
    - Periodically in monitoring (monthly/quarterly/yearly) to detect drift or deterioration.
    - After portfolio shifts, macro changes, or underwriting policy changes.
    """)

# -------------------------
# Tab 2: Tests & When to Use
# -------------------------
with tabs[1]:
    st.header("Which test to use and when")
    st.markdown("""
    **1) Exact Binomial Test (single grade)**  
    - Use when you test a *single* grade's PD: H₀: observed defaults ~ Binomial(n, PD_model).  
    - Good for small to moderate n; exact p-values.  
    - Example: PD=1%, n=200, observe 8 defaults — use binomial exact test.

    **2) Jeffreys Interval (Bayesian)**  
    - Jeffreys (Beta(0.5,0.5)) prior gives a conservative credible interval around observed default rate.  
    - Use for small samples or when you want an interval-based decision: check if model PD falls within the credible interval.  
    - More stable than Clopper-Pearson in the extremes.

    **3) Chi-square Goodness-of-Fit (multi-grade / portfolio-level)**  
    - Use when comparing **observed defaults across multiple grades** to expected defaults (n_i * PD_i).  
    - Requires sufficiently large expected counts (rule of thumb: all expected >= 5).  
    - If expected counts are small, prefer exact/binomial or aggregate bins.

    **Notes & practical tips**
    - Always consider sample size and power: failing to reject H₀ may be due to low power.
    - Statistical significance should be combined with business judgment — a small but economically material miscalibration may still require action.
    """)

# -------------------------
# Tab 3: Interactive Tests
# -------------------------
with tabs[2]:
    st.header("Interactive Calibration Testing")

    st.markdown("Choose test type and provide data. You can test single grades (binomial + Jeffreys) or multiple grades (chi-square).")

    test_mode = st.radio("Mode:", options=["Single-grade (exact binomial + Jeffreys)", "Multi-grade (Chi-square goodness-of-fit)"])

    if test_mode.startswith("Single"):
        st.subheader("Single-grade calibration test")
        col1, col2 = st.columns(2)
        with col1:
            n = st.number_input("Number of obligors (n)", min_value=1, value=200, step=1)
            defaults = st.number_input("Observed defaults (k)", min_value=0, max_value=1000000, value=5, step=1)
        with col2:
            pd_model = st.number_input("Model PD (as %)", min_value=0.0, max_value=100.0, value=2.0, step=0.01)
            alpha = st.number_input("Significance level α", min_value=0.001, max_value=0.2, value=0.05, step=0.001)
            alternative = st.selectbox("Alternative hypothesis for binomial test", options=["two-sided", "greater (observed > expected)", "less (observed < expected)"])
        alt_map = {"two-sided":"two-sided", "greater (observed > expected)":"greater", "less (observed < expected)":"less"}
        if st.button("Run single-grade tests"):
            p0 = pd_model/100.0
            k = int(defaults)
            # Binomial exact test
            p_binom = binomial_test_exact(k, int(n), p0, alternative=alt_map[alternative])
            # Jeffreys interval
            lower_j, upper_j = jeffreys_interval(k, int(n), alpha=alpha)
            # Observed rate
            obs_rate = k / n
            st.subheader("Results")
            st.write(f"Model PD = {p0:.4%}  •  Observed default rate = {obs_rate:.4%} ({k}/{n})")
            st.markdown("---")
            st.write("Exact binomial test")
            st.write(f"p-value = {p_binom:.4f}")
            st.write(interpret_pvalue(p_binom, alpha=alpha))
            st.markdown("---")
            st.write("Jeffreys interval (credible interval with Jeffreys prior)")
            st.write(f"{100*(1-alpha):.1f}% credible interval = [{lower_j:.4%}, {upper_j:.4%}]")
            if (p0 >= lower_j) and (p0 <= upper_j):
                st.success("Model PD falls inside the Jeffreys credible interval → no evidence of mis-calibration (interval check).")
            else:
                st.error("Model PD falls outside the Jeffreys credible interval → possible mis-calibration.")
            st.markdown("---")
            st.write("Quick interpretation guide:")
            st.write("- If binomial p-value < α → evidence model PD differs from observed frequency.")
            st.write("- If model PD outside Jeffreys interval → observed rate is inconsistent with model PD at the chosen credibility level.")
            # Plot
            fig, ax = plt.subplots()
            ax.bar(["Observed rate", "Model PD"], [obs_rate, p0], zorder=3)
            ax.errorbar([0], [obs_rate], yerr=[[obs_rate - lower_j], [upper_j - obs_rate]], fmt='o', capsize=8)
            ax.set_ylim(0, max(p0, obs_rate)*2 + 0.01)
            ax.set_ylabel("Default rate")
            ax.set_title("Observed vs Model PD (with Jeffreys interval)")
            st.pyplot(fig)

    else:
        st.subheader("Multi-grade calibration (Chi-square goodness-of-fit)")
        st.markdown("Enter grade counts, model PDs and observed defaults. Expected defaults = n_i * PD_i.")

        # Editable sample table
        default_sample = pd.DataFrame({
            "Grade": ["A", "B", "C", "D"],
            "Exposure": [1000, 800, 400, 200],
            "Model_PD_%": [0.5, 1.2, 3.0, 7.0],
            "Observed_Defaults": [5, 10, 16, 20]
        })
        st.write("Edit the sample dataset below (double-click cells to edit):")
        df = st.experimental_data_editor(default_sample, num_rows="fixed")
        # Ensure numeric types
        df['Exposure'] = df['Exposure'].astype(int)
        df['Model_PD_%'] = df['Model_PD_%'].astype(float)
        df['Observed_Defaults'] = df['Observed_Defaults'].astype(int)

        alpha = st.number_input("Significance level α", min_value=0.001, max_value=0.2, value=0.05, step=0.001)

        if st.button("Run chi-square calibration test"):
            exposures = df['Exposure'].tolist()
            obs = df['Observed_Defaults'].tolist()
            model_pds = (df['Model_PD_%'].values / 100.0).tolist()
            chi2_stat, pval_chi, expected_defaults = chi_square_calibration_test(obs, exposures, model_pds)

            st.subheader("Chi-square test results")
            if np.isnan(chi2_stat):
                st.error("Chi-square test not applicable: some expected default counts equal 0. Consider aggregating bins or adjusting PDs.")
            else:
                st.write(f"Chi-square statistic = {chi2_stat:.3f}")
                st.write(f"p-value = {pval_chi:.4f}")
                st.write(interpret_pvalue(pval_chi, alpha=alpha))
                st.write("Traffic-light:", traffic_light_from_p(pval_chi, alpha=alpha))
                st.markdown("Observed vs expected defaults (per grade):")
                table = pd.DataFrame({
                    "Grade": df['Grade'],
                    "Exposure": exposures,
                    "Model_PD_%": df['Model_PD_%'],
                    "Expected_Defaults": np.round(expected_defaults, 3),
                    "Observed_Defaults": obs,
                    "Observed_Rate_%": np.round(np.array(obs)/np.array(exposures)*100, 3)
                })
                st.table(table)

                # Check small expected counts
                small_expected = np.sum(expected_defaults < 5)
                if small_expected > 0:
                    st.warning(f"{small_expected} grade(s) have expected defaults < 5. Chi-square may be unreliable; consider aggregating grades or use exact methods.")

                # bar chart of observed vs expected defaults
                fig2, ax2 = plt.subplots()
                idx = np.arange(len(df))
                width = 0.35
                ax2.bar(idx - width/2, expected_defaults, width, label='Expected')
                ax2.bar(idx + width/2, df['Observed_Defaults'], width, label='Observed')
                ax2.set_xticks(idx)
                ax2.set_xticklabels(df['Grade'])
                ax2.set_ylabel("Defaults")
                ax2.set_title("Observed vs Expected defaults by grade")
                ax2.legend()
                st.pyplot(fig2)

# -------------------------
# Tab 4: Worked Examples
# -------------------------
with tabs[3]:
    st.header("Worked Examples (step-by-step)")

    st.subheader("Example A — Single grade, small sample (binomial + Jeffreys)")
    st.markdown("Model PD = 1.5%, n = 200 obligors, observed defaults = 8.")
    if st.button("Show Example A calculations"):
        k = 8
        n = 200
        p0 = 0.015
        p_bin = binomial_test_exact(k, n, p0, alternative='two-sided')
        lower_j, upper_j = jeffreys_interval(k, n, alpha=0.05)
        obs_rate = k / n
        st.write(f"Observed rate = {obs_rate:.4%}")
        st.write(f"Exact binomial p-value = {p_bin:.6f}")
        st.write(f"Jeffreys 95% interval = [{lower_j:.4%}, {upper_j:.4%}]")
        st.write("Interpretation:")
        st.write("- Binomial p-value small → evidence observed defaults exceed model PD." if p_bin < 0.05 else "- Binomial p-value not small → no strong evidence of mis-calibration.")
        st.write("- If model PD (1.5%) is outside Jeffreys interval, that's another signal of mis-calibration.")
        st.markdown("---")
        st.latex(r"\text{Binomial test: } H_0: X \sim \text{Binomial}(n, p_0)")
        st.latex(r"\text{Jeffreys interval: Posterior } \mathrm{Beta}(k+0.5, n-k+0.5)")

    st.subheader("Example B — Multi-grade chi-square")
    st.markdown("Grades A-D with model PDs and observed defaults (toy example).")
    if st.button("Show Example B calculations"):
        df_ex = pd.DataFrame({
            "Grade": ["A","B","C","D"],
            "Exposure": [1000, 800, 400, 200],
            "Model_PD_%": [0.5, 1.2, 3.0, 7.0],
            "Observed_Defaults": [5, 10, 16, 20]
        })
        st.table(df_ex)
        exposures = df_ex['Exposure'].tolist()
        obs = df_ex['Observed_Defaults'].tolist()
        model_pds = (df_ex['Model_PD_%'] / 100.0).tolist()
        chi2_stat, pval_chi, expected_defaults = chi_square_calibration_test(obs, exposures, model_pds)
        st.write(f"Chi-square = {chi2_stat:.3f}, p-value = {pval_chi:.4f}")
        st.write("Expected defaults:", np.round(expected_defaults,3))
        st.write("If p-value < 0.05 → evidence portfolio-level mis-calibration.")

# -------------------------
# Tab 5: Monitoring / Time Series
# -------------------------
with tabs[4]:
    st.header("Calibration Monitoring over time (demo)")
    st.markdown("""
    Monitor expected vs observed default rates over multiple observation windows (e.g., quarterly/annual).
    The demo below simulates or uses user-provided series to show how drift appears.
    """)

    st.subheader("Simulate a monitoring series (or paste your own)")
    col1, col2 = st.columns(2)
    with col1:
        periods = st.number_input("Number of periods (e.g., years)", min_value=2, max_value=24, value=6, step=1)
        base_pd = st.number_input("Base model PD (annual %, same for each period)", min_value=0.0, max_value=100.0, value=2.0, step=0.01)
    with col2:
        base_n = st.number_input("Exposure per period (n)", min_value=10, value=1000, step=1)
        drift = st.number_input("Annual drift in observed rate (%) (positive -> worse)", min_value=-100.0, max_value=100.0, value=0.5, step=0.01)

    if st.button("Simulate monitoring series"):
        # create series: observed rate = base_pd + drift * t + noise
        t = np.arange(periods)
        model_pd = base_pd / 100.0
        observed_rates = np.clip(model_pd + (drift/100.0) * t + np.random.normal(0, model_pd*0.2, size=periods), 0, 1)
        observed_defaults = np.random.binomial(base_n, observed_rates)
        expected_defaults = np.full(periods, base_n * model_pd)

        df_mon = pd.DataFrame({
            "Period": [f"T{int(i)+1}" for i in t],
            "Model_PD_%": np.round(model_pd*100, 4),
            "Observed_Defaults": observed_defaults,
            "Exposure": np.full(periods, base_n),
            "Observed_Rate_%": np.round(observed_defaults / base_n * 100, 4)
        })
        st.table(df_mon)

        # run binomial test each period
        pvals = [binomial_test_exact(int(df_mon.loc[i,'Observed_Defaults']), int(df_mon.loc[i,'Exposure']), model_pd, alternative='two-sided') for i in range(periods)]
        df_mon['Binomial_pvalue'] = np.round(pvals, 4)
        df_mon['Traffic'] = df_mon['Binomial_pvalue'].apply(lambda pv: traffic_light_from_p(pv, 0.05))

        st.write("Monitoring results with binomial tests per period:")
        st.table(df_mon[['Period','Model_PD_%','Observed_Rate_%','Binomial_pvalue','Traffic']])

        # plot observed vs model PD over time
        fig3, ax3 = plt.subplots()
        ax3.plot(df_mon['Period'], df_mon['Observed_Rate_%']/100.0, marker='o', label='Observed rate')
        ax3.plot(df_mon['Period'], df_mon['Model_PD_%']/100.0, marker='x', label='Model PD')
        ax3.set_ylabel("Default rate")
        ax3.set_title("Monitoring: observed vs model PD over time")
        ax3.legend()
        st.pyplot(fig3)

        st.markdown("""
        **Interpretation:** Repeated significant binomial test results or a clear upward trend in observed vs model PD indicates model deterioration or a shift in the portfolio — require investigation and possible recalibration.
        """)

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.subheader("Notes & Caveats")
st.markdown("""
- **Exact binomial** is robust for single-grade exact testing.  
- **Jeffreys interval** is a Bayesian credible interval using Beta(0.5,0.5) prior — often preferred for small samples because it's less conservative at extremes than Clopper-Pearson.  
- **Chi-square** requires reasonable expected counts (rule of thumb: expected >= 5). If expected counts are small, aggregate bins, or use exact/binomial approaches.
- **Statistical vs practical significance:** a tiny p-value could be statistically significant but economically immaterial — always combine with business judgement.
- **Multiple testing:** monitoring many grades and periods may require multiplicity adjustment (Bonferroni/Holm) to control false discovery rate.
""")

st.write("If you'd like, I can: (A) provide this app as a downloadable file, (B) add multiplicity-corrected pairwise post-hoc tests, or (C) include more advanced exact/Monte-Carlo tests for small expected counts. Which would you like?")
