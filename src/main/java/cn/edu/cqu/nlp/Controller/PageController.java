package cn.edu.cqu.nlp.Controller;

import org.springframework.stereotype.Controller;
import org.springframework.web.bind.annotation.RequestMapping;

@Controller
public class PageController {

	@RequestMapping(value = "/nlp")
	public String index() {
		return "index";
	}
}
