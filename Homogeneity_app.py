# homogeneity_app.py
import streamlit as st
import numpy as np
import pandas as pd
from math import sqrt
from scipy import stats
import matplotlib.pyplot as plt

st.set_page_config(page_title="Homogeneity vs Heterogeneity in Rating Grades", layout="wide")

# -------------------------
# Helper statistical funcs
# -------------------------
def two_proportion_z_test(count1, n1, count2, n2, alternative='two-sided'):
    """
    Returns z_stat, p_value, pooled_prop, p1, p2
    alternative: 'two-sided', 'larger' (p1>p2), 'smaller' (p1<p2)
    """
    # proportions
    p1 = count1 / n1 if n1 > 0 else 0
    p2 = count2 / n2 if n2 > 0 else 0
    pooled = (count1 + count2) / (n1 + n2) if (n1 + n2) > 0 else 0
    se = sqrt(pooled * (1 - pooled) * (1/n1 + 1/n2))
    # handle edge cases
    if se == 0:
        z = 0.0
        pval = 1.0
    else:
        z = (p1 - p2) / se
        if alternative == 'two-sided':
            pval = 2 * (1 - stats.norm.cdf(abs(z)))
        elif alternative == 'larger':  # p1 > p2
            pval = 1 - stats.norm.cdf(z)
        else:  # 'smaller' p1 < p2
            pval = stats.norm.cdf(z)
    return z, pval, pooled, p1, p2

def chi2_test_2xk(defaults, totals):
    """
    defaults: list/array of default counts per grade
    totals: list/array of total obligors per grade
    Builds contingency table: rows = default / non-default, columns = grades
    Uses scipy.stats.chi2_contingency
    Returns chi2, p, dof, expected, observed_table
    """
    defaults = np.array(defaults)
    totals = np.array(totals)
    non_defaults = totals - defaults
    table = np.vstack([defaults, non_defaults])
    chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)
    return chi2, p, dof, expected, table

def fisher_exact_test(count1, n1, count2, n2, alternative='two-sided'):
    """
    Fisher's exact test only for 2x2 contingency tables.
    Table:
        [[count1, n1-count1],
         [count2, n2-count2]]
    alternative: 'two-sided', 'greater', 'less' (scipy uses 'two-sided','greater','less')
    """
    table = [[int(count1), int(n1 - count1)], [int(count2), int(n2 - count2)]]
    # scipy's fisher_exact returns (oddsratio, pvalue)
    oddsratio, pvalue = stats.fisher_exact(table, alternative=alternative)
    return oddsratio, pvalue, table

# -------------------------
# UI - Title and Description
# -------------------------
st.title("Homogeneity vs Heterogeneity Between Rating Grades — Interactive App")
st.markdown("""
**Purpose:** Teach the difference between *homogeneous* and *heterogeneous* rating grades, 
show how to test homogeneity using proportions tests (z-test, chi-square, Fisher), and demonstrate why combining grades with different default behavior harms risk models.
""")

tabs = st.tabs(["Concepts", "Statistical Tests (Proportions)", "Worked Examples", "Practice: Combining Grades", "Multi-grade Homogeneity Test"])

# -------------------------
# Tab 1 - Concepts
# -------------------------
with tabs[0]:
    st.header("Conceptual Foundations: Homogeneity vs Heterogeneity")
    st.markdown("""
    **Homogeneity**  
    - A group is *homogeneous* when members are similar with respect to the characteristic of interest (e.g., probability of default).  
    - For rating grades, homogeneity means obligors assigned to the same grade have similar default rates → PD estimate for that grade is meaningful.

    **Heterogeneity**  
    - A group is *heterogeneous* when members differ in the characteristic of interest.  
    - If a rating grade is heterogeneous, a single PD estimate will poorly represent members inside the grade — leading to biased risk estimates.

    **Real-life analogies**
    - *Homogeneous basket:* 10 red apples, all roughly same size — you can say "the apples are similar".  
    - *Heterogeneous basket:* apples + oranges + bananas — a single average weight or price is misleading.
    """)
    st.subheader("Why it matters in risk modelling")
    st.markdown("""
    - **PD Estimation:** Homogeneous grades lead to stable, unbiased PD estimates.  
    - **Discrimination & Monotonicity:** Heterogeneous grades can break monotonic ordering of PDs across grades and reduce discrimination between grades.  
    - **Regulatory & Business Impact:** Poor grouping may fail validation/backtesting and lead to incorrect capital or pricing decisions.
    """)
    st.info("Tip: Homogeneity is *within* grade similarity. Heterogeneity is *between* grade differences that may be large enough to matter.")

# -------------------------
# Tab 2 - Statistical Tests
# -------------------------
with tabs[1]:
    st.header("Statistical Tests for Proportions (Interactive)")
    st.markdown("Use the widgets below to enter counts for two grades and run tests that check if default rates differ meaningfully.")

    st.subheader("Input counts for Grade A and Grade B")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Grade A**")
        n1 = st.number_input("Total obligors (n₁)", min_value=1, value=200, step=1)
        d1 = st.number_input("Defaults (x₁)", min_value=0, value=10, max_value=int(n1), step=1)
    with col2:
        st.markdown("**Grade B**")
        n2 = st.number_input("Total obligors (n₂)", min_value=1, value=200, step=1)
        d2 = st.number_input("Defaults (x₂)", min_value=0, value=20, max_value=int(n2), step=1)

    st.markdown("---")
    st.subheader("Choose test(s) to run")
    test_z = st.checkbox("Two-proportion z-test (approx parametric)", value=True)
    test_chi2 = st.checkbox("Chi-square test (2x2)", value=True)
    test_fisher = st.checkbox("Fisher's exact test (2x2, non-parametric)", value=False)
    alternative = st.radio("Alternative hypothesis (for z & Fisher):", options=["two-sided", "larger (p₁>p₂)", "smaller (p₁<p₂)"])
    alt_map = {'two-sided':'two-sided','larger (p₁>p₂)':'larger','smaller (p₁<p₂)':'smaller'}

    if st.button("Run tests"):
        st.markdown("### Results")
        results = {}
        if test_z:
            z_stat, pval_z, pooled, p1, p2 = two_proportion_z_test(int(d1), int(n1), int(d2), int(n2), alternative=alt_map[alternative])
            results['z'] = (z_stat, pval_z, pooled, p1, p2)
            st.subheader("Two-proportion z-test")
            st.latex(r"z = \frac{\hat p_1 - \hat p_2}{\sqrt{\hat p (1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")
            st.write(f"p̂₁ = {p1:.4f}, p̂₂ = {p2:.4f}, pooled p̂ = {pooled:.4f}")
            st.write(f"z-statistic = {z_stat:.3f}")
            st.write(f"p-value = {pval_z:.4f}")
            if pval_z < 0.05:
                st.success("Result: statistically significant difference (reject H₀ at 5%) → Evidence of heterogeneity.")
            else:
                st.info("Result: no significant difference (fail to reject H₀ at 5%) → Not enough evidence of heterogeneity.")
            st.markdown("---")

        if test_chi2:
            chi2, pchi, dof, expected, observed = chi2_test_2xk([int(d1), int(d2)], [int(n1), int(n2)])
            st.subheader("Chi-square test (2×2 contingency)")
            st.latex(r"\chi^2 = \sum \frac{(O - E)^2}{E}")
            st.write("Observed table (rows: defaults, non-defaults):")
            obs_df = pd.DataFrame(observed, index=["Defaults","Non-defaults"], columns=["Grade A","Grade B"])
            st.table(obs_df)
            st.write("Expected counts under H₀ (homogeneous default rates):")
            exp_df = pd.DataFrame(np.round(expected,3), index=["Defaults","Non-defaults"], columns=["Grade A","Grade B"])
            st.table(exp_df)
            st.write(f"Chi-square = {chi2:.3f}, df = {dof}, p-value = {pchi:.4f}")
            if pchi < 0.05:
                st.error("Result: statistically significant heterogeneity across grades (reject H₀).")
            else:
                st.success("Result: no significant heterogeneity detected (fail to reject H₀).")
            st.markdown("---")

        if test_fisher:
            odds, pfisher, table = fisher_exact_test(int(d1), int(n1), int(d2), int(n2), alternative=alt_map[alternative])
            st.subheader("Fisher's exact test (2×2)")
            st.write("Contingency table used:")
            st.table(pd.DataFrame(table, index=["Grade A","Grade B"], columns=["Defaults","Non-defaults"]))
            st.write(f"Odds ratio ≈ {odds if odds != float('inf') else 'inf'}")
            st.write(f"p-value = {pfisher:.4f}")
            if pfisher < 0.05:
                st.error("Result: significant difference (heterogeneity).")
            else:
                st.success("Result: no significant difference detected.")
            st.markdown("---")

    # small visualization
    st.subheader("Visualization of default rates")
    df_vis = pd.DataFrame({
        "Grade": ["Grade A", "Grade B"],
        "Defaults": [int(d1), int(d2)],
        "Total": [int(n1), int(n2)]
    })
    df_vis['Default Rate'] = df_vis['Defaults'] / df_vis['Total']
    fig, ax = plt.subplots()
    ax.bar(df_vis['Grade'], df_vis['Default Rate'])
    ax.set_ylim(0, max(0.1, df_vis['Default Rate'].max()*1.4))
    ax.set_ylabel("Default Rate")
    ax.set_title("Observed default rates")
    st.pyplot(fig)

# -------------------------
# Tab 3 - Worked Examples
# -------------------------
with tabs[2]:
    st.header("Worked Examples — Step-by-step")
    st.markdown("Below are small numeric examples showing the math and interpretation.")

    st.subheader("Example 1: Two-proportion z-test (manual calculation)")
    st.markdown("Scenario: Grade X has 8 defaults out of 160 obligors; Grade Y has 18 defaults out of 160 obligors.")
    if st.button("Show example 1 calculations"):
        x1, n1_ex = 8, 160
        x2, n2_ex = 18, 160
        z, pval, pooled, p1, p2 = two_proportion_z_test(x1, n1_ex, x2, n2_ex)
        st.write(f"p̂₁ = {p1:.4f} = {x1}/{n1_ex}")
        st.write(f"p̂₂ = {p2:.4f} = {x2}/{n2_ex}")
        st.write(f"pooled p̂ = {pooled:.4f} = {(x1+x2)}/{(n1_ex+n2_ex)}")
        st.latex(r"z = \frac{\hat p_1 - \hat p_2}{\sqrt{\hat p (1-\hat p)\left(\frac{1}{n_1}+\frac{1}{n_2}\right)}}")
        st.write(f"z = {z:.3f}")
        st.write(f"two-sided p-value = {pval:.4f}")
        if pval < 0.05:
            st.error("Interpretation: Significant difference in default rates → heterogeneity.")
        else:
            st.success("Interpretation: No significant difference detected.")

    st.markdown("---")
    st.subheader("Example 2: Chi-square for 3 grades (multi-column example)")
    st.markdown("Scenario (toy):")
    st.write("Grade A: 5 defaults / 100; Grade B: 12 defaults / 100; Grade C: 20 defaults / 100")
    if st.button("Show example 2 calculations"):
        defaults = [5, 12, 20]
        totals = [100, 100, 100]
        chi2, p, dof, expected, observed = chi2_test_2xk(defaults, totals)
        st.write("Observed table (defaults / non-defaults):")
        st.table(pd.DataFrame(observed, index=["Defaults","Non-defaults"], columns=["A","B","C"]))
        st.write("Expected counts under H₀ (equal default rate across grades):")
        st.table(pd.DataFrame(np.round(expected,3), index=["Defaults","Non-defaults"], columns=["A","B","C"]))
        st.write(f"Chi-square = {chi2:.3f}, df = {dof}, p-value = {p:.4f}")
        if p < 0.05:
            st.error("Interpretation: Evidence of heterogeneity across the 3 grades.")
        else:
            st.success("Interpretation: No evidence to reject homogeneity.")

# -------------------------
# Tab 4 - Practice: Combining Grades
# -------------------------
with tabs[3]:
    st.header("Practice Problem — Combining Grades & Why It Can Introduce Heterogeneity")
    st.markdown("""
    **Scenario:** A risk team considers merging *Grade B* and *Grade C* into a single grade because sample sizes are small.
    We'll show how different default rates make this merge problematic.
    """)

    st.subheader("Adjust counts for Grade B and Grade C")
    gb_n = st.number_input("Grade B total (n_B)", min_value=10, value=120, step=1, key='gb_n')
    gb_d = st.number_input("Grade B defaults (d_B)", min_value=0, value=6, max_value=int(gb_n), step=1, key='gb_d')
    gc_n = st.number_input("Grade C total (n_C)", min_value=10, value=80, step=1, key='gc_n')
    gc_d = st.number_input("Grade C defaults (d_C)", min_value=0, value=12, max_value=int(gc_n), step=1, key='gc_d')

    st.markdown("**Current default rates:**")
    rb = gb_d/gb_n if gb_n else 0
    rc = gc_d/gc_n if gc_n else 0
    st.write(f"Grade B default rate = {rb:.4f} ({gb_d}/{gb_n})")
    st.write(f"Grade C default rate = {rc:.4f} ({gc_d}/{gc_n})")

    if st.button("Assess merging B + C"):
        # test between B and C
        z_bc, pz_bc, *_ = two_proportion_z_test(int(gb_d), int(gb_n), int(gc_d), int(gc_n))
        chi2_bc, pchi_bc, dof_bc, expected_bc, observed_bc = chi2_test_2xk([int(gb_d), int(gc_d)], [int(gb_n), int(gc_n)])
        # merged
        merged_n = gb_n + gc_n
        merged_d = gb_d + gc_d
        merged_rate = merged_d / merged_n if merged_n else 0
        st.subheader("Test results (B vs C)")
        st.write(f"Two-proportion z-test: z = {z_bc:.3f}, p-value = {pz_bc:.4f}")
        st.write(f"Chi-square test: chi2 = {chi2_bc:.3f}, p-value = {pchi_bc:.4f}")
        if pz_bc < 0.05 or pchi_bc < 0.05:
            st.error("Conclusion: B and C have significantly different default rates → merging will produce a heterogeneous grade.")
        else:
            st.success("Conclusion: No strong statistical evidence that B and C differ (but consider power & sample sizes).")

        st.markdown("**If merged**")
        st.write(f"Merged Grade (B+C): defaults = {merged_d}, total = {merged_n}, merged default rate = {merged_rate:.4f}")

        st.subheader("Why merging heterogeneous grades harms models")
        st.markdown("""
        - **PD miscalibration:** The merged grade PD is an average weighted by counts; subgroups inside may have very different true PDs.  
        - **Loss of discriminatory power:** If one subgroup is much riskier, the merged grade masks the difference and model discrimination drops.  
        - **Monotonicity & ordering issues:** Merging can violate monotonic PD ordering across grades, complicating scorecard or rating logic.  
        - **Backtesting instability:** When sub-population exposures shift over time, the merged PD may swing and fail validation.
        """)
        st.info("Recommendation: Only merge if test shows no meaningful heterogeneity *and* business/validation analyses confirm homogeneity (consider sample size, stability).")

# -------------------------
# Tab 5 - Multi-grade homogeneity test
# -------------------------
with tabs[4]:
    st.header("Simple Homogeneity Test Across Multiple Rating Grades")
    st.markdown("Load the sample dataset, select grades to compare, and run a chi-square test across grades.")

    # sample dataset
    st.subheader("Sample dataset (toy)")
    sample_df = pd.DataFrame({
        "Grade": ["A", "B", "C", "D", "E"],
        "Total": [200, 150, 120, 80, 50],
        "Defaults": [2, 8, 12, 20, 15]
    })
    st.write("Default sample dataset (you can edit counts):")
    edited = st.experimental_data_editor(sample_df, num_rows="fixed")  # editable table

    grades = edited['Grade'].tolist()
    totals = edited['Total'].astype(int).tolist()
    defaults = edited['Defaults'].astype(int).tolist()

    st.subheader("Select grades to include in test")
    selected = st.multiselect("Choose grades", options=grades, default=grades)
    if len(selected) < 2:
        st.warning("Select at least two grades to perform a homogeneity test.")
    else:
        # filter
        sel_idx = [grades.index(g) for g in selected]
        sel_totals = [totals[i] for i in sel_idx]
        sel_defaults = [defaults[i] for i in sel_idx]

        st.write("Selected grades and counts:")
        st.table(pd.DataFrame({
            "Grade": selected,
            "Total": sel_totals,
            "Defaults": sel_defaults,
            "Default Rate": [f"{sel_defaults[i]/sel_totals[i]:.4f}" if sel_totals[i] else "NA" for i in range(len(sel_totals))]
        }))

        if st.button("Run multi-grade chi-square homogeneity test"):
            chi2_m, p_m, dof_m, expected_m, observed_m = chi2_test_2xk(sel_defaults, sel_totals)
            st.write(f"Chi-square = {chi2_m:.3f}, df = {dof_m}, p-value = {p_m:.4f}")
            st.write("Observed table (defaults / non-defaults):")
            st.table(pd.DataFrame(observed_m, index=["Defaults","Non-defaults"], columns=selected))
            st.write("Expected counts under homogeneity (defaults / non-defaults):")
            st.table(pd.DataFrame(np.round(expected_m,3), index=["Defaults","Non-defaults"], columns=selected))
            if p_m < 0.05:
                st.error("Result: Evidence of heterogeneity across selected grades (reject H₀).")
            else:
                st.success("Result: No evidence to reject homogeneity across selected grades.")

            st.subheader("Visual: default rates")
            df_plot = pd.DataFrame({
                "Grade": selected,
                "Default Rate": [sel_defaults[i]/sel_totals[i] if sel_totals[i] else 0 for i in range(len(sel_totals))]
            }).set_index('Grade')
            fig2, ax2 = plt.subplots()
            df_plot.plot(kind='bar', legend=False, ax=ax2)
            ax2.set_ylim(0, max(0.1, df_plot['Default Rate'].max()*1.3))
            ax2.set_ylabel("Default Rate")
            st.pyplot(fig2)

            st.markdown("**Why this test is useful:**")
            st.markdown("""
            - The chi-square test checks whether the observed differences in default counts are larger than expected by random sampling, assuming a common underlying default probability.  
            - If rejected → at least one grade behaves differently (heterogeneous).  
            - Next steps: investigate which grades differ (post-hoc tests, pairwise comparisons), consider revising grade definitions or splitting/merging strategically.
            """)

# -------------------------
# Footer / notes
# -------------------------
st.markdown("---")
st.markdown("**Notes & caveats**")
st.markdown("""
- Two-proportion z-test is an *approximate* test (large-sample). Use Fisher's exact test for small counts.  
- Chi-square across multiple grades requires expected counts not too small (rule of thumb: expected >= 5); otherwise consider exact or Monte Carlo methods.  
- Statistical significance is not the only criterion—consider business judgement, economic rationale, and stability when altering rating grades.
""")
st.write("If you'd like, I can (A) produce this as a single downloadable Python file, (B) add more advanced tests (e.g., post-hoc pairwise tests with multiplicity correction), or (C) convert the visuals to interactive Plotly charts. Which would you prefer?")
