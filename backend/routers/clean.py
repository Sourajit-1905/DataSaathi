from fastapi import APIRouter
router = APIRouter(prefix="/api/clean", tags=["clean"])

# "negative_values": ["keep", "set_zero", "set_nan", "drop_rows", "abs_value", "flag_only"],