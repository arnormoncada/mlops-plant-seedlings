import pandas as pd
from sklearn import datasets
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset, TargetDriftPreset
from evidently.test_suite import TestSuite
from evidently.tests import TestNumberOfMissingValues, TestColumnsType, TestNumberOfDuplicatedColumns, TestNumberOfEmptyRows, TestNumberOfColumns

reference_data = pd.read_csv('data_drifting/reference_database.csv')

current_data = pd.read_csv('data_drifting/prediction_database.csv')


reference_data = reference_data.drop(columns='Idx')
reference_data['Target'] = reference_data['Target'].astype('int64')
print(reference_data.head())
print(current_data.head())
# print(current_data.columns)

# print(current_data)
# current_data = current_data.replace({'NULL': -5})
# print(current_data)

report = Report(metrics=[DataDriftPreset(), DataQualityPreset(), TargetDriftPreset()])
report.run(reference_data=reference_data, current_data=current_data)
report.save_html('data_drifting/data_drift_metrics.html')


data_test = TestSuite(tests=[TestNumberOfMissingValues(),
                             TestColumnsType(),
                             TestNumberOfDuplicatedColumns(),
                             TestNumberOfEmptyRows(),
                             TestNumberOfColumns()])
data_test.run(reference_data=reference_data, current_data=current_data)
result = data_test.as_dict()
print(result)
print("All tests passed: ", result['summary']['all_passed'])

data_test.save_html('data_drifting/data_test.html')
