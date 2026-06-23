from ydata_profiling import ProfileReport
import pandas as pd 


df = pd.read_csv("train.csv")
report = ProfileReport(df, title="Pandas Profiling Report")
report.to_file("report.html")