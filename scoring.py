import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

class ParticipantVisibleError(Exception):
    pass

TARGET = ['x', 'y']

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Compute RMSE for NFL competition.
    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'id': unique identifier for each (game_id, play_id, nfl_id, frame_id)
      - Column 'x'
      - Column 'y'
    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = 'id'
    >>> solution = pd.DataFrame({'id': ['21_12_2_1', '21_12_2_2', '21_12_2_3'], 'x': [1,2,3], 'y':[4,2,3]})
    >>> submission  = pd.DataFrame({'id': ['21_12_2_1', '21_12_2_2', '21_12_2_3'], 'x': [1.1,2,3], 'y':[4,2.2,3]})
    >>> round(score(solution, submission, row_id_column_name=row_id_column_name), 4)
    0.0913
    >>> submission  = pd.DataFrame({'id': ['21_12_2_1', '21_12_2_2', '21_12_2_3'], 'x': [0,2,3], 'y':[4,2.2,3]})
    >>> round(score(solution, submission, row_id_column_name=row_id_column_name), 4)
    0.4163
    >>> submission  = pd.DataFrame({'id': ['21_12_2_1', '21_12_2_2', '21_12_2_3'], 'x': [1,2,1], 'y':[4,0,3]})
    >>> round(score(solution, submission, row_id_column_name=row_id_column_name), 4)
    1.1547
    """

    if row_id_column_name not in solution.columns:
        raise ParticipantVisibleError(f"Solution file missing required column: '{row_id_column_name}'")
    if row_id_column_name not in submission.columns:
        raise ParticipantVisibleError(f"Submission file missing required column: '{row_id_column_name}'")

    missing_in_solution = set(TARGET) - set(solution.columns)
    missing_in_submission = set(TARGET) - set(submission.columns)

    if missing_in_solution:
        raise ParticipantVisibleError(f'Solution file missing required columns: {missing_in_solution}')
    if missing_in_submission:
        raise ParticipantVisibleError(f'Submission file missing required columns: {missing_in_submission}')

    submission = submission[['id'] + TARGET]
    merged_df = pd.merge(solution, submission, on=row_id_column_name, suffixes=('_true', '_pred'))
    rmse = np.sqrt(
        0.5 * (mean_squared_error(merged_df['x_true'], merged_df['x_pred']) + mean_squared_error(merged_df['y_true'], merged_df['y_pred']))
    )
    return float(rmse)