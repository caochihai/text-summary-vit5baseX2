from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
import processingdata as prsdt
import summary as sum
import json

app = FastAPI()
app.config = {"JSON_AS_ASCII": False}

model = sum.Summary()

@app.post("/summarize_pdf")
async def summarize_pdf(pdf_path: str = Body(...)):
    try:
        # Kiểm tra đầu vào
        if not pdf_path:
            raise HTTPException(status_code=400, detail="Thiếu trường 'pdf_path' trong nội dung yêu cầu.")

        data_obj = prsdt.data(pdf_path)
        tt1, tt2 = data_obj.load_pdf()
        if tt1 and tt2:
            dt = data_obj.read_text()
            summary = model.summary_content(dt)

            return JSONResponse(
                content={"summary": summary},
                media_type="application/json; charset=utf-8"
            )

        elif not tt1 and not tt2:
            raise HTTPException(
                status_code=400,
                detail=f"Tệp PDF '{pdf_path}' không tồn tại hoặc không thể truy cập. "
                       f"Vui lòng kiểm tra lại đường dẫn hoặc quyền truy cập của tệp."
            )

        else:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Tệp PDF có thể là file scan ảnh hoặc tài liệu bằng ngôn ngữ không hỗ trợ. "
                    "Hiện hệ thống chỉ xử lý các file PDF có nội dung văn bản có thể sao chép được. "
                    "Vui lòng kiểm tra lại file và đảm bảo định dạng phù hợp."
                )
            )

    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Đã xảy ra lỗi trong quá trình xử lý yêu cầu.",
                "chi_tiet": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)