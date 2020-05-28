package cn.edu.cqu.nlp.Dao;

import cn.edu.cqu.nlp.Model.Record;
import cn.edu.cqu.nlp.Model.RecordExample;
import cn.edu.cqu.nlp.Model.RecordWithBLOBs;
import java.util.List;

import org.apache.ibatis.annotations.Mapper;
import org.apache.ibatis.annotations.Param;

@Mapper
public interface RecordMapper {
	long countByExample(RecordExample example);

	int deleteByExample(RecordExample example);

	int deleteByPrimaryKey(Integer recordId);

	int insert(RecordWithBLOBs record);

	int insertSelective(RecordWithBLOBs record);

	List<RecordWithBLOBs> selectByExampleWithBLOBs(RecordExample example);

	List<Record> selectByExample(RecordExample example);

	RecordWithBLOBs selectByPrimaryKey(Integer recordId);

	int updateByExampleSelective(@Param("record") RecordWithBLOBs record, @Param("example") RecordExample example);

	int updateByExampleWithBLOBs(@Param("record") RecordWithBLOBs record, @Param("example") RecordExample example);

	int updateByExample(@Param("record") Record record, @Param("example") RecordExample example);

	int updateByPrimaryKeySelective(RecordWithBLOBs record);

	int updateByPrimaryKeyWithBLOBs(RecordWithBLOBs record);

	int updateByPrimaryKey(Record record);

	Integer getId();
}