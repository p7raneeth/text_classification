    features, target = columns[:-1], columns[-1]
    percent_num_vals = round(df.isnull().sum()/df.shape[0],5)*100
    return {"data": df.iloc[:num_records, :],
           "percent_null_values": percent_num_vals,
           "features": features,
           "target": target}