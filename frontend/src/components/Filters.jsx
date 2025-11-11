import React, { useState } from 'react';

const Filters = ({ onFilterChange }) => {
  const [filters, setFilters] = useState({
    search: '',
    category: '',
    priority: '',
  });

  const handleChange = (field, value) => {
    const newFilters = { ...filters, [field]: value };
    setFilters(newFilters);
    onFilterChange(newFilters);
  };

  const clearFilters = () => {
    const emptyFilters = { search: '', category: '', priority: '' };
    setFilters(emptyFilters);
    onFilterChange(emptyFilters);
  };

  return (
    <div className="filters">
      <div className="filter-group">
        <label>Search</label>
        <input
          type="text"
          placeholder="Search emails..."
          value={filters.search}
          onChange={(e) => handleChange('search', e.target.value)}
        />
      </div>

      <div className="filter-group">
        <label>Category</label>
        <select value={filters.category} onChange={(e) => handleChange('category', e.target.value)}>
          <option value="">All Categories</option>
          <option value="work">Work</option>
          <option value="personal">Personal</option>
          <option value="urgent">Urgent</option>
          <option value="promotion">Promotion</option>
        </select>
      </div>

      <div className="filter-group">
        <label>Priority</label>
        <select value={filters.priority} onChange={(e) => handleChange('priority', e.target.value)}>
          <option value="">All Priorities</option>
          <option value="high">High</option>
          <option value="medium">Medium</option>
          <option value="low">Low</option>
        </select>
      </div>

      <div className="filter-group" style={{ justifyContent: 'flex-end' }}>
        <label>&nbsp;</label>
        <button className="btn btn-secondary" onClick={clearFilters}>
          Clear Filters
        </button>
      </div>
    </div>
  );
};

export default Filters;
