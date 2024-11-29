// Inicialize o mapa e defina uma localização inicial e zoom
const map = L.map('map').setView([-23.5505, -46.6333], 6); // São Paulo como exemplo

L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
attribution: '&copy; OpenStreetMap contributors',
}).addTo(map);

let marker;

map.on('click', function (e) {
const { lat, lng } = e.latlng;

// Se já existe um marcador, remova-o
if (marker) {
    map.removeLayer(marker);
}

// Adicione um marcador na localização clicada
marker = L.marker([lat, lng]).addTo(map);

latitude.value = lat.toFixed(6); // Limite para 6 casas decimais
longitude.value = lng.toFixed(6);

// Exiba as coordenadas (ex.: console ou input)
console.log(`Latitude: ${lat}, Longitude: ${lng}`);
});