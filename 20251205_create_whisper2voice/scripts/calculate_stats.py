"""Calculate statistics for analysis results."""
import pandas as pd


def main() -> None:
    """Calculate and print statistics for both datasets."""
    # whisper_converted_v2
    df1 = pd.read_csv('results/metrics_whisper_converted_v2.csv')
    print('whisper_converted_v2')
    print('MCD_dB', round(df1['MCD_dB'].mean(), 4),
          round(df1['MCD_dB'].std(), 4))
    print('Mel_MSE', round(df1['Mel_MSE'].mean(), 4),
          round(df1['Mel_MSE'].std(), 4))
    print('F0_MSE', round(df1['F0_MSE'].mean(), 4),
          round(df1['F0_MSE'].std(), 4))
    print('LogF0_RMSE', round(df1['LogF0_RMSE'].mean(), 4),
          round(df1['LogF0_RMSE'].std(), 4))
    print('SP_MSE', round(df1['SP_MSE'].mean(), 4),
          round(df1['SP_MSE'].std(), 4))
    print('AP_MSE', round(df1['AP_MSE'].mean(), 4),
          round(df1['AP_MSE'].std(), 4))

    # whisper10
    df2 = pd.read_csv('results/metrics_whisper10.csv')
    print('whisper10')
    print('MCD_dB', round(df2['MCD_dB'].mean(), 4),
          round(df2['MCD_dB'].std(), 4))
    print('Mel_MSE', round(df2['Mel_MSE'].mean(), 4),
          round(df2['Mel_MSE'].std(), 4))
    print('F0_MSE', round(df2['F0_MSE'].mean(), 4),
          round(df2['F0_MSE'].std(), 4))
    print('LogF0_RMSE', round(df2['LogF0_RMSE'].mean(), 4),
          round(df2['LogF0_RMSE'].std(), 4))
    print('SP_MSE', round(df2['SP_MSE'].mean(), 4),
          round(df2['SP_MSE'].std(), 4))
    print('AP_MSE', round(df2['AP_MSE'].mean(), 4),
          round(df2['AP_MSE'].std(), 4))


if __name__ == "__main__":
    main()
