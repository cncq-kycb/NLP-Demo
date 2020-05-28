package cn.edu.cqu.nlp.Model;

public class MyJson {
	private boolean success;
	private String message;
	private Object data;

	public boolean isSuccess() {
		return success;
	}

	public void setSuccess(boolean success) {
		this.success = success;
	}

	public String getMessage() {
		return message;
	}

	public void setMessage(String message) {
		this.message = message;
	}

	public Object getData() {
		return data;
	}

	public void setData(Object data) {
		this.data = data;
	}

	public MyJson(boolean success, String message) {
		this.success = success;
		this.message = message;
	}

	public MyJson(Object data) {
		super();
		this.success = true;
		this.data = data;
	}
}
