const apiKey = 'ae948b4627b7974dc7686b83b973dac6';

const form = document.getElementById('formid'); 

form.addEventListener('submit', async function(event){
    event.preventDefault();

    // Mostrar a área de resultado
    if(document.querySelector('.container-resultado')){
        document.querySelector('.container-resultado').classList.remove('invisivel');
    }

    // Coletar latitude e longitude
    const latitude = document.getElementById('latitude').value;
    const longitude = document.getElementById('longitude').value;

    try {
        // Fazer requisição à API do OpenWeather para obter a previsão de tempo
        const response = await fetch(`https://api.openweathermap.org/data/2.5/forecast?lat=${latitude}&lon=${longitude}&appid=${apiKey}&units=metric`);
        const data = await response.json();
        
        // Extrair dados necessários para a predição (pegando o primeiro intervalo de previsão)
        const temperatura = data.list[0].main.temp; // Temperatura em Celsius (primeira previsão)
        const precipitacao = data.list[0].rain ? 1 : 0; // Verifica se tem chuva (0 = não, 1 = sim)

        // Enviar esses dados para a API do Flask para fazer a predição
        const predictionResponse = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                latitude: parseFloat(latitude),
                longitude: parseFloat(longitude),
                precipitacao: precipitacao,
                temperatura: temperatura,
                 // Exemplo fixo, pode ser alterado conforme a entrada
            })
        });

        const predictionData = await predictionResponse.json();
        console.log(predictionData);

        // Processar a predição recebida
        if (predictionData.error) {
            throw new Error(predictionData.error);
        }

        // Exibir o resultado com base na predição
        calcularRiscoDeFogo(predictionData.prediction[0]);

    } catch (error) {
        console.error('Erro na requisição ou na predição:', error);
    }
});

// Função para exibir o gráfico de risco
function calcularRiscoDeFogo(result){
    if ( document.querySelector('.container-resultado')) {
        document.querySelector('.container-resultado').remove();
    }

    const container = document.createElement('div');
    container.className = 'container-resultado';
    container.innerHTML = `
        <h3 class="risco-text"></h3>
        <div class="grafico-risco">
            <canvas id="myChart"></canvas>
        </div>
    `;
    document.body.appendChild(container);

    window.scrollTo({
        top: container.offsetTop,
        behavior: 'smooth'
    });

    let barColor;
    let classificacao;
    if (result < 3) {
        barColor = 'green';
        classificacao = 'Risco Baixo';
    } else if (result >= 3 && result < 6) {
        barColor = 'yellow';
        classificacao = 'Risco Moderado';
    } else {
        barColor = 'red';
        classificacao = 'Risco Alto';
    }

    document.querySelector('.risco-text').innerHTML = classificacao;

    new Chart(document.getElementById('myChart'), {
        type: 'bar',
        data: {
            labels: [classificacao],
            datasets: [{
                label: 'Risco de Fogo',
                data: [result],
                borderWidth: 1,
                backgroundColor: barColor
            }]
        },
        options: {
            scales: {
                y: {
                    beginAtZero: true,
                    min: 0,
                    max: 10,
                    stepSize: 1,
                }
            }
        }
    });
}
