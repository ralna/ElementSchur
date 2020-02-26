import pandas as pd


def combine_tables(tables, DoF, columns, formatters):
    data_frames = {}
    for name, table_dict in tables.items():

        cols = pd.MultiIndex.from_product([[name], [u'Time', u'Iteration']])
        df = pd.DataFrame(data=table_dict)

        for col_name, col in df.iteritems():
            df[col_name] = col.apply(lambda x: formatters[col_name].format(x))

        df.columns = cols
        data_frames[name] = df

    concat_results = [data_frames[key] for key in data_frames.keys()]
    final_table = pd.concat([pd.DataFrame(data={"DoF": DoF})] + concat_results,
                            axis=1, sort=False)
    final_table["DoF"] = final_table["DoF"].apply(lambda x: f"{x:5.0f}")
    return final_table
