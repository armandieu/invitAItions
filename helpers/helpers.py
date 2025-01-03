import pandas as pd
def feature_engin(df):
    if 'action_start_date' in df.columns:
        df['action_start_date'] = pd.to_datetime(df['action_start_date'])

    if 'action_end_date' in df.columns:
        df['action_end_date'] = pd.to_datetime(df['action_end_date'])

    if 'transaction_date' in df.columns:
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])
    spend_by_category = df.groupby(['client_id', 'product_category_2']).agg(
        total_gross_amount = ('gross_amount_euro', 'sum'),
        total_quantity = ('product_quantity', 'sum'),
        mean_gross_amount = ('gross_amount_euro', 'mean'),
        mean_quantity = ('product_quantity', 'mean')
    ).reset_index()
    # Calculate total gross amount and total quantity for each 'client_id', 'year', and 'month'
    spent_total = df.groupby(['client_id']).agg(
        total_gross_amount = ('gross_amount_euro', 'sum'),
        total_quantity = ('product_quantity', 'sum'),
        mean_gross_amount = ('gross_amount_euro', 'mean'),
        mean_quantity = ('product_quantity', 'mean')
    ).reset_index()
    pivot_df = spend_by_category.pivot_table(index=['client_id'], columns='product_category_2', fill_value=0)
    # Flatten the multi-level columns
    pivot_df.columns = [f'{col[0]}_{col[1]}' for col in pivot_df.columns]
    # Reset the index to make 'client_id', 'year', and 'month' as regular columns
    pivot_df.reset_index(inplace=True)
    agg_transactions = df.groupby('client_id').agg(
        num_transactions=('client_id', 'size'),
        customer_life_months=('transaction_date', lambda x: (x.max() - x.min()).days / 30), 
        top_prod_cat=('product_category', lambda x: x.mode().iloc[0]),
        top_prod_subcat=('product_subcategory', lambda x: x.mode().iloc[0]),
    ).reset_index()
    final_df = pd.merge(pivot_df, spent_total, on=['client_id']).merge(agg_transactions, on=['client_id'])
    return final_df