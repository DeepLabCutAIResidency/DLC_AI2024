import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px

from src.data import *
from src.calcs import *
from src.utils import *
from src.visualisation import *


data = pd.read_csv(
    "/home/dikra/MyHub/Code/DLC24_Hub/DLC_AI2024/analysis_pipeline/predicted_clusters.csv",
    index_col=False,
)

app = dash.Dash(
    __name__,
    assets_folder="/home/dikra/media/dikra/PhD/DATA/DLC24_Data/tiny_all_bird_merged_coco/images",
)

app.layout = html.Div(
    [
        dcc.Graph(id="scatter-plot"),
        html.Div(id="image-container", style={"width": "10px", "height": "10px"}),
    ]
)


@app.callback(
    Output("image-container", "children"), [Input("scatter-plot", "hoverData")]
)
def update_image(hoverData):
    if hoverData is None:
        return html.Img(src="")
    else:
        image_path = hoverData["points"][0]["customdata"][0]
        return html.Img(src=app.get_asset_url(image_path.split("/")[-1]))


fig = px.scatter(
    data, x="0", y="1", color="label", hover_data=["image_path", "src_dataset"]
)
app.layout.children[0].figure = fig


if __name__ == "__main__":
    app.run_server(debug=True)
