from diagrams import Diagram, Edge
from diagrams.aws.compute import EC2
from diagrams.aws.database import RDS
from diagrams.generic.device import Mobile
from diagrams.aws.mobile import APIGateway
from diagrams.aws.storage import SimpleStorageServiceS3Bucket
from diagrams.aws.compute import Lambda

from diagrams import Cluster, Diagram

graph_attr = {
    "fontsize": "20"
}

with Diagram("Data Collection", show=True,graph_attr=graph_attr):

    mobile = Mobile("Samsung Galaxy Note 10 Plus")

    with Cluster("AWS Cloud"):
        aws = APIGateway("Upload endpoint") 
        aws - Lambda("Store file") >> SimpleStorageServiceS3Bucket("Raw data storage") >> Edge(label="Object created event")>> Lambda("Audio segmentation") >> [SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files"),SimpleStorageServiceS3Bucket("Store individual audio files")]

    mobile >> Edge(label="Audio File Data") >> aws