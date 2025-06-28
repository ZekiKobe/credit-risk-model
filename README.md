### Credit Scoring Business Understanding.

Task 1: Credit Scoring Business Understanding
Here's what you should include in your README.md:

How Basel II influences model requirements
Basel II emphasizes three pillars: minimum capital requirements, supervisory review, and market discipline. This means:

Our model must be interpretable to satisfy regulatory scrutiny

Documentation must clearly show how risk is measured and quantified

We need to demonstrate the model's predictive power and stability over time

The model must align with the bank's risk appetite and capital allocation strategies

Proxy variable necessity and risks
Since we lack direct default data:

A proxy based on RFM (Recency, Frequency, Monetary) patterns can estimate risk

Potential proxy: Customers with frequent late payments or fraud flags as "high risk"

Business risks include:

Misclassification leading to lost revenue (false positives) or defaults (false negatives)

Proxy may not perfectly correlate with actual repayment behavior

Potential bias in the proxy affecting certain customer segments

Model complexity trade-offs
Simple models (Logistic Regression with WoE):

Pros: Easily interpretable, regulatory-friendly, simpler to validate

Cons: May miss complex patterns, lower predictive power

Complex models (Gradient Boosting):

Pros: Higher accuracy, captures non-linear relationships

Cons: Black-box nature raises regulatory concerns, harder to explain decisions

In regulated finance, we often start simple and only add complexity if it provides material improvement that justifies the compliance overhead.