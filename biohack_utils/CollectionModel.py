from pydantic import BaseModel, Field, model_validator
from typing import Optional, Literal, Union, Annotated

class LabelInfo(BaseModel):
    """Minimal label metadata - just identifies source."""
    source: Optional[str] = None  # Node name of source image in same collection

class OMEROInfo(BaseModel):
    """OMERO identifiers for round-tripping."""
    image_id: int
    dataset_id: Optional[int] = None

class NodeAttributes(BaseModel):
    """Node attributes - extensible."""
    label: Optional[LabelInfo] = None
    omero: Optional[OMEROInfo] = None
    
    class Config:
        extra = "allow"

class MultiscaleNode(BaseModel):
    """An OME-Zarr image (intensity or label)."""
    type: Literal["multiscale"] = "multiscale"
    name: str
    path: Optional[str] = None  # None when only in OMERO, set on export
    attributes: Optional[NodeAttributes] = None
    
    def is_label(self) -> bool:
        return self.attributes is not None and self.attributes.label is not None

class CollectionNode(BaseModel):
    """Groups nodes together."""
    type: Literal["collection"] = "collection"
    name: str
    nodes: Optional[list[Union["CollectionNode", MultiscaleNode]]] = None
    path: Optional[str] = None
    attributes: Optional[dict] = None  # For plate/well metadata etc.
    
    @model_validator(mode="after")
    def check_nodes_or_path(self):
        if self.nodes is not None and self.path is not None:
            raise ValueError("Cannot have both 'nodes' and 'path'")
        return self

CollectionNode.model_rebuild()

Node = Annotated[
    Union[CollectionNode, MultiscaleNode],
    Field(discriminator="type")
]

class OMECollection(BaseModel):
    """Root collection."""
    version: str = "0.x"
    type: Literal["collection"] = "collection"
    name: str
    nodes: list[Node]
    attributes: Optional[dict] = None

class OMEWrapper(BaseModel):
    """Top-level 'ome' wrapper."""
    ome: OMECollection