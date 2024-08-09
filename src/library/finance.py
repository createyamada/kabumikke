import yfinance as yf
from library import format



def get_company_info(code_ :str):
    """
    特定コードの企業情報を取得する

    Parameters:
    - code_:str 株価コード
    Returns:
    - result: object 企業情報
    """

    return yf.Ticker(code_+".T")

    # # 分析に必要なデータフレームを配列で取得
    # analysis = get_analysis_data(code_)

    # 個別のデータフレームを1つのデータフレームにまとめデータを整形する
    # return format.merge_all_company_info(analysis)


def get_analysis_data(company):
    """
    分析のためのデータを取得する

    Parameters:
    - company:str 特定の企業データ(yfinanceで取得)

    Returns:
    - result: 全分析データフレームの配列
    """
    result = [];

    # # 企業情報を取得
    # company = yf.Ticker(code_+".T")

    # 企業の株価時系列
    result.append(company.history(period="max"))

    # 配当金及び株式分割の確定日を取得
    result.append(company.actions)

    # # 財務諸表直近四年分
    result.append(company.financials)

    # # 財務諸表直近四半期分取得
    result.append(company.quarterly_financials)

    # # バランスシート直近4年分
    result.append(company.balance_sheet)

    # # バランスシート直近四半期分
    result.append(company.quarterly_balance_sheet)

    # 個別のデータフレームを1つのデータフレームにまとめデータを整形する
    return format.merge_all_company_info(result)

    


