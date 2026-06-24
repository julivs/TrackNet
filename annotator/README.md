# TrackNet · Anotador de Bola

Ferramenta web para anotação manual de posição de bola em vídeos, gerando o `Label.csv` necessário para fine-tuning do [TrackNet](https://arxiv.org/abs/1907.03698).

## Requisitos

- [Node.js](https://nodejs.org/) 18+
- [ffmpeg](https://ffmpeg.org/) instalado e disponível no PATH

Verifique:
```bash
node --version
ffmpeg -version
```

## Instalação

```bash
cd annotator
npm install
```

## Uso

### 1. Inicie o servidor

```bash
npm start
```

O terminal exibirá os endereços de acesso:
```
TrackNet · Anotador
  Local : http://localhost:3000
  LAN   : http://192.168.x.x:3000
```

Qualquer dispositivo na mesma rede pode acessar via o endereço LAN.

### 2. Coloque os vídeos

Copie os vídeos de beach tennis para a pasta `../videos/` (raiz do repositório):

```
tracknet/
  videos/
    partida1.mp4
    partida2.mp4
```

Formatos suportados: `.mp4`, `.mov`, `.avi`, `.mkv`, `.webm`

### 3. Extraia os frames

No browser, selecione o vídeo e defina o **FPS de anotação**.

> **Importante:** use o mesmo FPS que será usado na inferência. O modelo usa 3 frames consecutivos para detectar o movimento da bola — o intervalo temporal entre eles deve ser consistente entre treino e inferência.
>
> O vídeo original geralmente é 30 fps. Use 30 fps para máxima fidelidade, ou um valor menor (ex: 10–15 fps) se quiser reduzir o volume de anotação e adaptar a inferência ao mesmo fps.

Clique em **Extrair frames** — o ffmpeg processa o vídeo no servidor e exibe o progresso em tempo real.

### 4. Anote os frames

| Ação | Resultado |
|------|-----------|
| Clique no frame | Marca posição da bola (`visibility=1`) |
| Botão **Ausente** ou `Espaço` | Marca frame sem bola (`visibility=0`) |
| `←` / `→` | Navega entre frames |
| `U` | Desfaz anotação do frame atual |
| Barra de progresso | Clique para navegar; verde = bola, vermelho = ausente |

Após marcar um frame, avança automaticamente para o próximo.

O progresso é salvo automaticamente no servidor (`sessions/<nome>/annotations.json`). Recarregar a página ou abrir de outro dispositivo retoma de onde parou.

### 5. Exporte o CSV

Clique em **Baixar CSV** a qualquer momento para obter o `Label.csv`.

Frames não anotados são exportados como `visibility=0` (ausente).

## Estrutura de saída

```
annotator/sessions/<nome_do_video>/
  frames/
    0000.jpg
    0001.jpg
    ...
  annotations.json   ← progresso salvo automaticamente
```

O `Label.csv` exportado segue o formato exato esperado pelo `gt_gen.py`:

```
file name,visibility,x-coordinate,y-coordinate,status
0000.jpg,0,,,0
0001.jpg,1,642,380,0
0002.jpg,1,655,371,0
```

## Próximos passos após anotar

1. Organize os frames e o CSV na estrutura do dataset:

```
datasets/trackNet/images/game1/Clip1/   ← copie os frames aqui
                          Clip1/Label.csv
```

2. Gere os ground truth heatmaps e os CSVs de treino/validação:

```bash
python gt_gen.py --path_input datasets/trackNet --path_output datasets/trackNet
```

3. Inicie o fine-tuning:

```bash
python main.py --exp_id beach_tennis
```
