import { useState } from 'react';
import { FaSearch, FaTimes, FaFilter } from 'react-icons/fa';
import '../styles/AdvancedSearch.css';

function AdvancedSearch({ onSearch, onClear }) {
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [filters, setFilters] = useState({
    q: '',
    categories: [],
    priorities: [],
    is_read: null,
    start_date: '',
    end_date: '',
    min_priority_score: ''
  });

  const categories = ['WORK', 'PERSONAL', 'URGENT', 'PROMOTION', 'OTHER'];
  const priorities = ['HIGH', 'MEDIUM', 'LOW'];

  const handleInputChange = (field, value) => {
    setFilters(prev => ({ ...prev, [field]: value }));
  };

  const handleMultiSelect = (field, value) => {
    setFilters(prev => {
      const current = prev[field];
      const updated = current.includes(value)
        ? current.filter(v => v !== value)
        : [...current, value];
      return { ...prev, [field]: updated };
    });
  };

  const handleSearch = () => {
    onSearch(filters);
  };

  const handleClear = () => {
    const clearedFilters = {
      q: '',
      categories: [],
      priorities: [],
      is_read: null,
      start_date: '',
      end_date: '',
      min_priority_score: ''
    };
    setFilters(clearedFilters);
    onClear();
  };

  return (
    <div className="advanced-search">
      <div className="search-bar">
        <FaSearch className="search-icon" />
        <input
          type="text"
          placeholder="Search emails by subject, sender, or content..."
          value={filters.q}
          onChange={(e) => handleInputChange('q', e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          className="search-input"
        />
        <button 
          onClick={() => setShowAdvanced(!showAdvanced)} 
          className="filter-toggle"
          title="Advanced filters"
        >
          <FaFilter /> {showAdvanced ? 'Hide' : 'Filters'}
        </button>
        {(filters.q || filters.categories.length > 0 || filters.priorities.length > 0) && (
          <button onClick={handleClear} className="clear-btn" title="Clear all filters">
            <FaTimes />
          </button>
        )}
      </div>

      {showAdvanced && (
        <div className="advanced-filters">
          <div className="filter-section">
            <label>Categories</label>
            <div className="checkbox-group">
              {categories.map(cat => (
                <label key={cat} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={filters.categories.includes(cat)}
                    onChange={() => handleMultiSelect('categories', cat)}
                  />
                  <span>{cat}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="filter-section">
            <label>Priorities</label>
            <div className="checkbox-group">
              {priorities.map(pri => (
                <label key={pri} className="checkbox-label">
                  <input
                    type="checkbox"
                    checked={filters.priorities.includes(pri)}
                    onChange={() => handleMultiSelect('priorities', pri)}
                  />
                  <span>{pri}</span>
                </label>
              ))}
            </div>
          </div>

          <div className="filter-section">
            <label>Read Status</label>
            <select 
              value={filters.is_read === null ? '' : filters.is_read}
              onChange={(e) => handleInputChange('is_read', e.target.value === '' ? null : e.target.value === 'true')}
              className="select-input"
            >
              <option value="">All</option>
              <option value="false">Unread</option>
              <option value="true">Read</option>
            </select>
          </div>

          <div className="filter-section">
            <label>Date Range</label>
            <div className="date-range">
              <input
                type="date"
                value={filters.start_date}
                onChange={(e) => handleInputChange('start_date', e.target.value)}
                className="date-input"
                placeholder="From"
              />
              <span>to</span>
              <input
                type="date"
                value={filters.end_date}
                onChange={(e) => handleInputChange('end_date', e.target.value)}
                className="date-input"
                placeholder="To"
              />
            </div>
          </div>

          <div className="filter-section">
            <label>Min Priority Score</label>
            <input
              type="number"
              step="0.1"
              min="0"
              max="1"
              value={filters.min_priority_score}
              onChange={(e) => handleInputChange('min_priority_score', e.target.value)}
              className="number-input"
              placeholder="0.0 - 1.0"
            />
          </div>

          <div className="filter-actions">
            <button onClick={handleSearch} className="btn-primary">
              <FaSearch /> Apply Filters
            </button>
            <button onClick={handleClear} className="btn-secondary">
              <FaTimes /> Clear All
            </button>
          </div>
        </div>
      )}
    </div>
  );
}

export default AdvancedSearch;
