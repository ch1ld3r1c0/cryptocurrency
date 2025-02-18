import yfinance as yf
import pandas as pd
from ta.trend import EMAIndicator, MACD
from ta.momentum import RSIIndicator
from textblob import TextBlob
import requests
from tabulate import tabulate
from groq import Groq
from time import sleep
import os
from ta.volatility import AverageTrueRange
from ta.volume import VolumeWeightedAveragePrice

# Função para chamar a API do Qwen
def call_qwen_api(texto: str) -> str:
    """Chama a API do Qwen para extrair recomendações técnicas com base nos scores."""
    query = f"Tendo os scores fornecidos abaixo, quais são as recomendações técnicas para cada ativo? Recomende se é o caso de compra ou venda no curto, médio ou longo prazo. Informe, também, se a compra ou venda do ativo é vantajoso ou não. Evite conclusões conflitantes. Responda em Português.\n\n{texto}"
    
    try:
        client = Groq(
            api_key="",
        )
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": f"""{query}""",
                }
            ],
            model="llama3-70b-8192",
        )
        
        return chat_completion.choices[0].message.content
    
    except Exception as e:
        print(e)
        return "Erro na recomendação pela API."

def fetch_crypto_data(ticker, period="6mo", interval="1d"):
    """Busca dados históricos do criptoativo usando Yahoo Finance."""
    data = yf.download(ticker, period=period, interval=interval)
    if isinstance(data.columns, pd.MultiIndex):
        ticker_column = data.columns.get_level_values(1)[0]
        return data, ticker_column
    return data, None

def fetch_sentiment_score(query):
    """Busca o sentimento geral para um ativo usando uma API ou modelo local."""
    try:
        # NewsAPI (insira sua chave de API abaixo)
        api_key = ""
        url = f"https://newsapi.org/v2/everything?q={query}&language=en&apiKey={api_key}"
        response = requests.get(url)
        response.raise_for_status()
        articles = response.json()["articles"]

        sentiments = []
        for article in articles:
            title = article["title"]
            analysis = TextBlob(title).sentiment
            sentiments.append(analysis.polarity)

        return sum(sentiments) / len(sentiments) if sentiments else 0
    except Exception as e:
        print(f"Erro ao buscar sentimento para {query}: {e}")
        return None


def analyze_trend(data, ticker_column):
    """Analisa a tendência do criptoativo usando indicadores técnicos."""
    try:
        close_data = data[('Close', ticker_column)] if ticker_column else data['Close']
        close_data = pd.to_numeric(close_data, errors='coerce').astype(float)

        if len(close_data) < 50:
            return False, False, False

        window_50 = min(50, len(close_data))
        window_200 = min(200, len(close_data))

        sma_50 = close_data.rolling(window=window_50).mean()
        sma_200 = close_data.rolling(window=window_200).mean()

        ema = EMAIndicator(close=close_data, window=min(20, len(close_data)))
        ema_20 = ema.ema_indicator()

        rsi = RSIIndicator(close=close_data, window=min(14, len(close_data)))
        rsi_values = rsi.rsi()

        # Análise de curto prazo
        short_term_buy = rsi_values.iloc[-1] < 45 and close_data.iloc[-1] > ema_20.iloc[-1]

        # Análise de médio prazo
        medium_term_buy = window_50 == 50 and sma_50.iloc[-1] > sma_200.iloc[-1]

        # Análise de longo prazo
        long_term_buy = False
        if window_200 == 200:
            # Condição 1: Preço atual está acima do SMA 200
            price_above_sma200 = close_data.iloc[-1] > sma_200.iloc[-1]

            # Condição 2: SMA 200 está em tendência de alta (inclinação positiva)
            sma_200_slope = (sma_200.iloc[-1] - sma_200.iloc[-50]) / sma_200.iloc[-50] * 100
            sma_200_trending_up = sma_200_slope > 0  # SMA 200 subindo

            # Condição 3: Margem de tolerância (opcional)
            margin = 0.02  # 2% acima do SMA 200
            price_above_with_margin = close_data.iloc[-1] > sma_200.iloc[-1] * (1 + margin)

            # Combinação das condições
            long_term_buy = price_above_sma200 and sma_200_trending_up and price_above_with_margin

        return short_term_buy, medium_term_buy, long_term_buy
    except Exception as e:
        return False, False, False

def analyze_sentiment(asset):
    """Analisa o sentimento de mercado para um criptoativo."""
    sentiment_score = fetch_sentiment_score(asset)
    if sentiment_score is None:
        return "Sem dados", 0
    if sentiment_score > 0.1:
        return "Compra", 3
    elif sentiment_score < -0.1:
        return "Venda", -3
    else:
        return "Neutro", 1

def analyze_fundamentals(data, ticker_column):
    """Analisa métricas fundamentais do ativo, como volume e market cap."""
    try:
        close_data = data[('Close', ticker_column)] if ticker_column else data['Close']
        volume_data = data[('Volume', ticker_column)] if ticker_column else data['Volume']

        price_change = ((close_data.iloc[-1] - close_data.iloc[-7]) / close_data.iloc[-7]) * 100
        avg_volume = volume_data.rolling(window=7).mean().iloc[-1]

        if price_change > 5 and avg_volume > volume_data.mean():
            return "Fundamentos Fortes", 3
        elif price_change < -5:
            return "Fundamentos Fracos", -3
        else:
            return "Neutro", 1
    except Exception as e:
        return "Neutro", 0

def aggressive_fundamental_analysis(data, ticker_column):
    """Realiza uma análise fundamentalista mais agressiva."""
    try:
        close_data = data[('Close', ticker_column)] if ticker_column else data['Close']
        volume_data = data[('Volume', ticker_column)] if ticker_column else data['Volume']

        price_change = ((close_data.iloc[-1] - close_data.iloc[-14]) / close_data.iloc[-14]) * 100
        volume_spike = volume_data.iloc[-1] > 1.5 * volume_data.rolling(window=14).mean().iloc[-1]

        if price_change > 10 and volume_spike:
            return "Alta demanda e grande valorização", 4
        elif price_change < -10:
            return "Grande desvalorização", -4
        else:
            return "Sem sinais claros", 0
    except Exception as e:
        return "Sem sinais claros", 0

def analyze_volatility(data, ticker_column):
    """Analisa a volatilidade do ativo usando o indicador Average True Range (ATR)."""
    try:
        high_data = data[('High', ticker_column)] if ticker_column else data['High']
        low_data = data[('Low', ticker_column)] if ticker_column else data['Low']
        close_data = data[('Close', ticker_column)] if ticker_column else data['Close']

        atr = AverageTrueRange(high=high_data, low=low_data, close=close_data, window=min(14, len(close_data)))
        atr_value = atr.average_true_range().iloc[-1]

        if atr_value > close_data.mean() * 0.05:  # ATR > 5% do preço médio
            return "Alta Volatilidade", 3
        else:
            return "Baixa Volatilidade", 1
    except Exception as e:
        return "Erro na análise de volatilidade", 0

def analyze_volume(data, ticker_column):
    """Analisa o volume de negociação do ativo."""
    try:
        volume_data = data[('Volume', ticker_column)] if ticker_column else data['Volume']
        avg_volume = volume_data.rolling(window=14).mean().iloc[-1]
        current_volume = volume_data.iloc[-1]

        if current_volume > 1.5 * avg_volume:
            return "Volume Alto", 3
        else:
            return "Volume Normal", 1
    except Exception as e:
        return "Erro na análise de volume", 0

def analyze_momentum(data, ticker_column):
    """Analisa o momentum do ativo usando o indicador MACD."""
    try:
        close_data = data[('Close', ticker_column)] if ticker_column else data['Close']
        macd = MACD(close=close_data, window_slow=26, window_fast=12, window_sign=9)
        macd_line = macd.macd()
        signal_line = macd.macd_signal()

        if macd_line.iloc[-1] > signal_line.iloc[-1]:
            return "Momentum Positivo", 3
        else:
            return "Momentum Negativo", -3
    except Exception as e:
        return "Erro na análise de momentum", 0

def analyze_support_resistance(data, ticker_column):
    """Identifica níveis de suporte e resistência."""
    try:
        close_data = data[('Close', ticker_column)] if ticker_column else data['Close']
        resistance = close_data.rolling(window=14).max().iloc[-1]
        support = close_data.rolling(window=14).min().iloc[-1]
        current_price = close_data.iloc[-1]

        if current_price > resistance:
            return "Rompeu Resistência", 3
        elif current_price < support:
            return "Rompeu Suporte", -3
        else:
            return "Entre Suporte e Resistência", 1
    except Exception as e:
        return "Erro na análise de suporte e resistência", 0

def analyze_liquidity(data, ticker_column):
    """Analisa a liquidez do ativo com base no volume e no spread."""
    try:
        volume_data = data[('Volume', ticker_column)] if ticker_column else data['Volume']
        high_data = data[('High', ticker_column)] if ticker_column else data['High']
        low_data = data[('Low', ticker_column)] if ticker_column else data['Low']

        spread = high_data.iloc[-1] - low_data.iloc[-1]
        avg_spread = (high_data - low_data).rolling(window=14).mean().iloc[-1]

        if volume_data.iloc[-1] > 10000 and spread < avg_spread * 0.5:
            return "Alta Liquidez", 3
        else:
            return "Baixa Liquidez", 1
    except Exception as e:
        return "Erro na análise de liquidez", 0

def rank_assets_with_qwen(assets):
    """Classifica os criptoativos por vantagem para compra e atualiza recomendações via API Qwen."""
    recommendations = []
    output_dir = os.path.join(os.getcwd(), "recomendacoes")
    os.makedirs(output_dir, exist_ok=True)

    for asset in assets:
        try:
            data, ticker_column = fetch_crypto_data(asset)

            if data.empty:
                recommendations.append({"Ativo": asset, "Recomendação": "Sem dados disponíveis."})
                continue

            short, medium, long = analyze_trend(data, ticker_column)
            sentiment, sentiment_score = analyze_sentiment(asset)
            fundamentals, fundamentals_score = analyze_fundamentals(data, ticker_column)
            aggressive_analysis, aggressive_score = aggressive_fundamental_analysis(data, ticker_column)
            volatility, volatility_score = analyze_volatility(data, ticker_column)
            volume, volume_score = analyze_volume(data, ticker_column)
            momentum, momentum_score = analyze_momentum(data, ticker_column)
            support_resistance, support_resistance_score = analyze_support_resistance(data, ticker_column)
            liquidity, liquidity_score = analyze_liquidity(data, ticker_column)

            scores_text = (
                f"Ativo: {asset}\n"
                f"Curto prazo: {short}\n"
                f"Médio prazo: {medium}\n"
                f"Longo prazo: {long}\n"
                f"Sentimento: {sentiment} ({sentiment_score})\n"
                f"Fundamentalista: {fundamentals} ({fundamentals_score})\n"
                f"Análise agressiva: {aggressive_analysis} ({aggressive_score})\n"
                f"Volatilidade: {volatility} ({volatility_score})\n"
                f"Volume: {volume} ({volume_score})\n"
                f"Momentum: {momentum} ({momentum_score})\n"
                f"Suporte/Resistência: {support_resistance} ({support_resistance_score})\n"
                f"Liquidez: {liquidity} ({liquidity_score})\n"
            )

            sleep(30)
            qwen_recommendation = call_qwen_api(scores_text)

            recommendations.append({"Ativo": asset, "Recomendação": qwen_recommendation})
            
            # Salva a recomendação em arquivo .txt
            file_name = f"{asset}.txt"
            file_path = os.path.join(output_dir, file_name)
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(f"Recomendações para {asset}:\n\n{scores_text}\n{qwen_recommendation}")
        except Exception as e:
            recommendations.append({"Ativo": asset, "Recomendação": f"Erro ao analisar {asset}: {e}"})
   
    return recommendations

def display_recommendations(recommendations):
    """Exibe recomendações em formato de tabela usando tabulate."""
    print("\n--- Recomendações de Compra ---")
    print(tabulate(recommendations, headers="keys", tablefmt="grid"))

def save_recommendations_to_files(recommendations, folder_name="recomendacoes"):
    """Salva as recomendações em arquivos .txt separados por ativo."""
    import os

    # Cria o diretório se não existir
    folder_path = os.path.join(os.getcwd(), "criptoativos", folder_name)
    os.makedirs(folder_path, exist_ok=True)

    for recommendation in recommendations:
        asset = recommendation["Ativo"]
        content = recommendation["Recomendação"]

        # Define o caminho do arquivo para o ativo
        file_name = f"{asset}.txt"
        file_path = os.path.join(folder_path, file_name)

        try:
            # Salva o conteúdo no arquivo
            with open(file_path, "w", encoding="utf-8") as file:
                file.write(f"Ativo: {asset}\n\nRecomendação:\n{content}\n")
        except Exception as e:
            print(f"Erro ao salvar recomendação para {asset}: {e}")

    print(f"\nRecomendações salvas na pasta: {folder_path}")

# Atualiza a execução principal para incluir o salvamento
if __name__ == "__main__":
    
    file_path = os.path.join(os.getcwd(), "ativos.txt")

    try:
        with open(file_path, "r") as file:
            crypto_assets = [line.strip() for line in file.readlines()]
    except FileNotFoundError:
        crypto_assets = ["FTM-USD"]

    recommendations = rank_assets_with_qwen(crypto_assets)
    display_recommendations(recommendations)
    save_recommendations_to_files(recommendations)



# Explicação das Melhorias
# Preço acima do SMA 200:

# price_above_sma200 = close_data.iloc[-1] > sma_200.iloc[-1]

# Isso verifica se o preço atual está acima da média de 200 dias, o que é um sinal clássico de tendência de alta de longo prazo.

# Inclinação do SMA 200:

# sma_200_slope = (sma_200.iloc[-1] - sma_200.iloc[-50]) / sma_200.iloc[-50] * 100

# Calcula a variação percentual do SMA 200 nos últimos 50 dias. Se for positiva, indica que a média está subindo.

# Margem de tolerância:

# price_above_with_margin = close_data.iloc[-1] > sma_200.iloc[-1] * (1 + margin)

# Adiciona uma margem de 2% para evitar falsos sinais em mercados laterais.

# Combinação das condições:

# long_term_buy = price_above_sma200 and sma_200_trending_up and price_above_with_margin

# Apenas retorna True se todas as condições forem atendidas.