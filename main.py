print("=" * 60)
print("             RIDERUSH PROJECT")
print("=" * 60)

print("\nStep 1: Data Preprocessing")
exec(open("01_preprocessing.py").read())

print("\nStep 2: Exploratory Data Analysis")
exec(open("02_eda.py").read())

print("\nStep 3: Demand Prediction - Linear Regression")
exec(open("03_demand_linear.py").read())

print("\nStep 4: ETA Prediction - Linear Regression")
exec(open("04_eta_linear.py").read())

print("\n" + "=" * 60)
print("          RIDERUSH COMPLETED")
print("=" * 60)